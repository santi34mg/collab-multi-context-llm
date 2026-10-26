//! CLI for native inference, local debug teachers, and sequence distillation.

use crate::distillation::{self, DistillPrompt};
use crate::experiments;
use crate::inference::http::{self, HttpBackend};
use crate::inference::{ChatMessage, GenerationConfig, InferenceBackend};
use crate::model::TinyTransformer;
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

pub const HELP: &str = "Local GGUF inference and teacher distillation

  infer -m MODEL.gguf -p PROMPT [-n 256] [--raw] [--json]
  infer --preset qwen|lfm -p PROMPT
  infer --endpoint http://127.0.0.1:8080 -p PROMPT [--teacher-id ID]
  serve -m MODEL.gguf [--host 127.0.0.1] [--port 8080] [--alias ID]
  distill --preset qwen|lfm --input PROMPTS.json --output SAMPLES.jsonl
  distill --endpoint http://127.0.0.1:8080 --input PROMPTS.json --output SAMPLES.jsonl
  train-distill --input SAMPLES.jsonl [--input OTHER.jsonl] --epochs N --output STUDENT.json
  student --checkpoint STUDENT.json -p PROMPT [--max-bytes 64]

Native model selection: --model/-m PATH or --preset qwen|lfm (requires native feature).
Native settings: --ctx-size/-c 2048 --batch-size/-b 128 --threads/-t 2
                 --n-gpu-layers/-ngl 0 [--thinking]
Generation: --max-tokens/-n 256 --temp 0.2 --top-p 0.95 --seed 42 [--stop TEXT ...]
HTTP debug settings: --endpoint URL [--teacher-id ID] [--timeout 300]
Inference: --prompt/-p TEXT or --file/-f PATH; optional --system TEXT.
Student training: --width 32 --ctx-size 256 --streams 2 --learning-rate 0.01
                  [--resume EXISTING_STUDENT.json]

Presets resolve under HOME/.local/share/local-models. No weights are downloaded.
HTTP is unauthenticated loopback-only, non-streaming, for trusted local debugging.
Teacher GGUF weights are frozen. train-distill updates only the small Rust student.
";

#[derive(Default)]
struct Flags(BTreeMap<String, Vec<String>>);

fn canonical(name: &str) -> &str {
    match name {
        "-m" => "model",
        "-p" => "prompt",
        "-f" => "file",
        "-n" | "--n-predict" => "max-tokens",
        "-c" => "ctx-size",
        "-b" => "batch-size",
        "-t" => "threads",
        "-ngl" | "--n-gpu-layers" => "gpu-layers",
        "-h" => "help",
        other => other.strip_prefix("--").unwrap_or(other),
    }
}

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut flags = Self::default();
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            if !arg.starts_with('-') {
                return Err(format!("expected an option, found {arg:?}"));
            }
            let name = canonical(arg).to_owned();
            index += 1;
            let value = if ["raw", "json", "thinking", "help"].contains(&name.as_str()) {
                "true".into()
            } else {
                let value = args
                    .get(index)
                    .ok_or_else(|| format!("missing value for {arg}"))?;
                if value.starts_with('-') && value.parse::<f64>().is_err() {
                    return Err(format!("missing value for {arg}"));
                }
                index += 1;
                value.clone()
            };
            flags.0.entry(name).or_default().push(value);
        }
        Ok(flags)
    }
    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn one(&mut self, name: &str) -> Result<Option<String>, String> {
        match self.0.remove(name) {
            None => Ok(None),
            Some(mut values) if values.len() == 1 => Ok(values.pop()),
            Some(_) => Err(format!("--{name} may only be specified once")),
        }
    }
    fn required(&mut self, name: &str) -> Result<String, String> {
        self.one(name)?
            .ok_or_else(|| format!("--{name} is required"))
    }
    fn flag(&mut self, name: &str) -> Result<bool, String> {
        self.one(name).map(|value| value.is_some())
    }
    fn many(&mut self, name: &str) -> Vec<String> {
        self.0.remove(name).unwrap_or_default()
    }
    fn number<T: FromStr>(&mut self, name: &str, default: T) -> Result<T, String> {
        self.one(name)?.map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| format!("invalid --{name}: {value:?}"))
        })
    }
    fn finish(self) -> Result<(), String> {
        match self.0.keys().next() {
            Some(name) => Err(format!("unknown or inapplicable option --{name}")),
            None => Ok(()),
        }
    }
}

struct BackendArgs {
    model: Option<PathBuf>,
    endpoint: Option<String>,
    teacher_id: Option<String>,
    timeout: Duration,
    context: u32,
    batch: u32,
    threads: i32,
    gpu_layers: i32,
    thinking: bool,
}

impl BackendArgs {
    fn parse(flags: &mut Flags) -> Result<Self, String> {
        let endpoint = flags.one("endpoint")?;
        let teacher_id = flags.one("teacher-id")?;
        let timeout = flags.number("timeout", 300u64)?;
        if timeout == 0 {
            return Err("--timeout must be positive".into());
        }
        let model = flags.one("model")?;
        let preset = flags.one("preset")?;
        if model.is_some() && preset.is_some() {
            return Err("choose --model or --preset, not both".into());
        }
        if endpoint.is_some()
            && (model.is_some()
                || preset.is_some()
                || [
                    "ctx-size",
                    "batch-size",
                    "threads",
                    "gpu-layers",
                    "thinking",
                ]
                .iter()
                .any(|name| flags.has(name)))
        {
            return Err("--endpoint cannot be combined with native model options; use --teacher-id for an HTTP model ID".into());
        }
        if endpoint.is_none() && teacher_id.is_some() {
            return Err("--teacher-id requires --endpoint".into());
        }
        let model = match (model, preset) {
            (Some(path), _) => Some(expand_path(&path)?),
            (_, Some(preset)) => {
                let file = match preset.as_str() {
                    "qwen" => "Qwen3.5-4B-Q5_K_M.gguf",
                    "lfm" => "LFM2.5-8B-A1B-Q4_K_M.gguf",
                    _ => return Err("--preset must be qwen or lfm".into()),
                };
                Some(expand_path(&format!("~/.local/share/local-models/{file}"))?)
            }
            _ => None,
        };
        if endpoint.is_none() && model.is_none() {
            return Err("specify --model, --preset, or --endpoint".into());
        }
        let context = flags.number("ctx-size", 2048u32)?;
        let batch = flags.number("batch-size", 128u32.min(context))?;
        let threads = flags.number("threads", 2i32)?;
        let gpu_layers = flags.number("gpu-layers", 0i32)?;
        let thinking = flags.flag("thinking")?;
        if context == 0 || batch == 0 || batch > context || threads <= 0 {
            return Err(
                "context, batch, and threads must be positive; batch must fit context".into(),
            );
        }
        Ok(Self {
            model,
            endpoint,
            teacher_id,
            timeout: Duration::from_secs(timeout),
            context,
            batch,
            threads,
            gpu_layers,
            thinking,
        })
    }

    fn open(self) -> Result<Box<dyn InferenceBackend>, String> {
        if let Some(endpoint) = self.endpoint {
            return Ok(Box::new(HttpBackend::connect(
                &endpoint,
                self.teacher_id,
                self.timeout,
            )?));
        }
        #[cfg(feature = "native")]
        {
            use crate::inference::native::{NativeConfig, NativeModel};
            let mut model = NativeModel::load(NativeConfig {
                model: self.model.expect("native model selection validated"),
                context_size: self.context,
                batch_size: self.batch,
                threads: self.threads,
                gpu_layers: self.gpu_layers,
            })?;
            model.set_thinking(self.thinking);
            Ok(Box::new(model))
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = (
                self.model,
                self.context,
                self.batch,
                self.threads,
                self.gpu_layers,
                self.thinking,
            );
            Err("native GGUF inference requires building with --features native and LLAMA_CPP_DIR pointing to the compatible llama.cpp library; --endpoint works without that feature".into())
        }
    }
}

fn expand_path(path: &str) -> Result<PathBuf, String> {
    if let Some(suffix) = path.strip_prefix("~/") {
        let home = std::env::var_os("HOME").ok_or("HOME is required to expand ~/ model paths")?;
        Ok(PathBuf::from(home).join(suffix))
    } else {
        Ok(PathBuf::from(path))
    }
}

fn generation(flags: &mut Flags) -> Result<GenerationConfig, String> {
    let options = GenerationConfig {
        max_tokens: flags.number("max-tokens", 256)?,
        temperature: flags.number("temp", 0.2)?,
        top_p: flags.number("top-p", 0.95)?,
        seed: flags.number("seed", 42)?,
        stop: flags.many("stop"),
    };
    options.validate()?;
    Ok(options)
}

fn messages(flags: &mut Flags) -> Result<Vec<ChatMessage>, String> {
    let prompt = flags.one("prompt")?;
    let file = flags.one("file")?;
    let prompt = match (prompt, file) {
        (Some(prompt), None) => prompt,
        (None, Some(path)) => {
            std::fs::read_to_string(&path).map_err(|e| format!("read prompt {path}: {e}"))?
        }
        _ => return Err("specify exactly one of --prompt or --file".into()),
    };
    let mut messages = Vec::new();
    if let Some(system) = flags.one("system")? {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(prompt));
    crate::inference::validate_messages(&messages)?;
    Ok(messages)
}

fn read_json<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    serde_json::from_reader(BufReader::new(
        File::open(path).map_err(|e| format!("open {path}: {e}"))?,
    ))
    .map_err(|e| format!("read {path}: {e}"))
}

struct AliasBackend {
    inner: Box<dyn InferenceBackend>,
    alias: String,
}
impl InferenceBackend for AliasBackend {
    fn model_id(&self) -> &str {
        &self.alias
    }
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        options: &GenerationConfig,
    ) -> Result<crate::inference::Generation, String> {
        let mut result = self.inner.generate(messages, options)?;
        result.model = self.alias.clone();
        Ok(result)
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    let name = args.first().ok_or("missing inference command")?;
    let mut flags = Flags::parse(&args[1..])?;
    if flags.flag("help")? {
        print!("{HELP}");
        return Ok(());
    }
    match name.as_str() {
        "infer" => {
            let backend = BackendArgs::parse(&mut flags)?;
            let options = generation(&mut flags)?;
            let messages = messages(&mut flags)?;
            let raw = flags.flag("raw")?;
            let json = flags.flag("json")?;
            if raw && (messages.len() != 1 || backend.endpoint.is_some()) {
                return Err("--raw requires a native model and no --system message".into());
            }
            flags.finish()?;
            let mut backend = backend.open()?;
            let answer = if raw {
                backend.complete_raw(&messages[0].content, &options)?
            } else {
                backend.generate(&messages, &options)?
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&answer).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", answer.text);
            }
            eprintln!(
                "{}: {} prompt tokens, {} completion tokens, finish={}",
                answer.model, answer.prompt_tokens, answer.completion_tokens, answer.finish_reason
            );
            Ok(())
        }
        "serve" => {
            let backend = BackendArgs::parse(&mut flags)?;
            if backend.endpoint.is_some() {
                return Err("serve requires a native model, not an HTTP proxy".into());
            }
            let host = flags.one("host")?.unwrap_or_else(|| "127.0.0.1".into());
            let port = flags.number("port", 8080u16)?;
            let max_requests = flags
                .one("max-requests")?
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| "invalid --max-requests".to_string())
                })
                .transpose()?;
            if max_requests == Some(0) {
                return Err("--max-requests must be positive".into());
            }
            let alias = flags.one("alias")?;
            if alias.as_ref().is_some_and(|a| a.trim().is_empty()) {
                return Err("--alias cannot be empty".into());
            }
            flags.finish()?;
            // Validate binding policy before loading gigabytes of model weights.
            if !["127.0.0.1", "localhost", "::1", "[::1]"].contains(&host.as_str()) {
                return Err("serve is restricted to loopback hosts".into());
            }
            let address = if host.contains(':') {
                format!("[{}]:{port}", host.trim_matches(['[', ']']))
            } else {
                format!("{host}:{port}")
            };
            let backend = backend.open()?;
            let mut backend: Box<dyn InferenceBackend> = match alias {
                Some(alias) => Box::new(AliasBackend {
                    inner: backend,
                    alias,
                }),
                None => backend,
            };
            eprintln!(
                "serving {} on http://{address}; loopback diagnostic API, no authentication or streaming",
                backend.model_id()
            );
            http::serve(backend.as_mut(), &address, max_requests)
        }
        "distill" => {
            let backend = BackendArgs::parse(&mut flags)?;
            let options = generation(&mut flags)?;
            let input = flags.required("input")?;
            let output = flags.required("output")?;
            flags.finish()?;
            if std::path::Path::new(&output)
                .try_exists()
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "dataset already exists: {output}; choose a new output path"
                ));
            }
            let prompts: Vec<DistillPrompt> = read_json(&input)?;
            let mut backend = backend.open()?;
            eprintln!(
                "collecting {} prompts from {}; local prompts and answers will be stored in {output}",
                prompts.len(),
                backend.model_id()
            );
            let samples = distillation::collect(backend.as_mut(), &prompts, &options)?;
            distillation::write_jsonl(&output, &samples)?;
            println!(
                "saved {} completed teacher answers to {output}",
                samples.len()
            );
            Ok(())
        }
        "train-distill" => {
            let inputs = flags.many("input");
            if inputs.is_empty() {
                return Err("at least one --input JSONL file is required".into());
            }
            let epochs = flags.number("epochs", 1usize)?;
            let rate = flags.number("learning-rate", 0.01f32)?;
            let output = flags
                .one("output")?
                .unwrap_or_else(|| "student.json".into());
            let resume = flags.one("resume")?;
            let dimension_flags = ["width", "ctx-size", "streams"]
                .iter()
                .any(|name| flags.has(name));
            let width = flags.number("width", 32usize)?;
            let positions = flags.number("ctx-size", 256usize)?;
            let streams = flags.number("streams", 2usize)?;
            if resume.is_some() && dimension_flags {
                return Err(
                    "--resume uses saved dimensions; omit --width, --ctx-size, and --streams"
                        .into(),
                );
            }
            flags.finish()?;
            let mut samples = Vec::new();
            for input in inputs {
                samples.extend(distillation::read_jsonl(input)?);
            }
            let mut model = match resume {
                Some(path) => TinyTransformer::load(path)?,
                None => {
                    if streams == 0 {
                        return Err("--streams must be positive".into());
                    }
                    let mut config = experiments::small_config();
                    config.width = width;
                    config.max_positions = positions;
                    config.max_streams = streams.checked_add(1).ok_or("stream count overflow")?;
                    TinyTransformer::new(config, 7)?
                }
            };
            let stats = distillation::train_samples(&mut model, &samples, epochs, rate)?;
            model.save(&output)?;
            println!(
                "{} samples; {} supervised answer bytes; in-sample loss {:.4} -> {:.4}; saved {output}",
                samples.len(),
                stats.trained_bytes,
                stats.initial_loss,
                stats.final_loss
            );
            Ok(())
        }
        "student" => {
            let checkpoint = flags.required("checkpoint")?;
            let messages = messages(&mut flags)?;
            let max_bytes = flags.number("max-bytes", 64usize)?;
            flags.finish()?;
            let model = TinyTransformer::load(checkpoint)?;
            println!(
                "{}",
                distillation::predict_answer(&model, &messages, max_bytes)?
            );
            Ok(())
        }
        _ => Err(format!("unknown inference command {name:?}\n{HELP}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn aliases_duplicates_and_negative_gpu_layers() {
        let mut flags =
            Flags::parse(&args(&["-m", "model.gguf", "-ngl", "-1", "-n", "8"])).unwrap();
        let backend = BackendArgs::parse(&mut flags).unwrap();
        assert_eq!(backend.gpu_layers, -1);
        assert_eq!(generation(&mut flags).unwrap().max_tokens, 8);
        flags.finish().unwrap();
        let mut flags = Flags::parse(&args(&["-m", "a", "--model", "b"])).unwrap();
        assert!(flags.one("model").is_err());
    }

    #[test]
    fn rejects_ambiguous_or_ignored_backend_settings() {
        for values in [
            vec!["--preset", "other"],
            vec!["-m", "a", "--preset", "qwen"],
            vec!["--endpoint", "http://localhost:8080", "-m", "a"],
            vec!["--endpoint", "http://localhost:8080", "--thinking"],
            vec!["--teacher-id", "qwen"],
        ] {
            assert!(BackendArgs::parse(&mut Flags::parse(&args(&values)).unwrap()).is_err());
        }
        assert!(Flags::parse(&args(&["--model", "--prompt", "x"])).is_err());
        assert!(
            run(&args(&[
                "infer",
                "--preset",
                "qwen",
                "-p",
                "hello",
                "--unknown",
                "x"
            ]))
            .unwrap_err()
            .contains("unknown")
        );
    }

    #[test]
    fn validates_generation_and_prompt_before_inference() {
        assert!(generation(&mut Flags::parse(&args(&["--temp", "NaN"])).unwrap()).is_err());
        assert!(
            messages(&mut Flags::parse(&args(&["-p", "hello", "-f", "other"])).unwrap()).is_err()
        );
        assert!(messages(&mut Flags::parse(&args(&["-p", ""])).unwrap()).is_err());
    }
}
