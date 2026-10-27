use collab_multi_context_llm::context::{self, ContextBudget};
use collab_multi_context_llm::experiments;
use collab_multi_context_llm::memory::{MemoryKind, MemoryStore};
use collab_multi_context_llm::model::TinyTransformer;
use collab_multi_context_llm::state::{Engine, EngineConfig, Operation, RoundReport, StreamId};
use collab_multi_context_llm::training::{self, TrainingEpisode};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

const HELP: &str = "Collaborative multi-context research prototype (CPU Rust)

Usage:
  collab-multi-context-llm demo [MEMORY_FILE] [TRACE_FILE]
      Scripted collaboration through the byte protocol, durable memory, restart.
  collab-multi-context-llm train [STEPS=12000] [CHECKPOINT=checkpoint.json]
      Train a tiny transformer to read the other stream's published A/B bit.
  collab-multi-context-llm train-trace DATASET EPOCHS [CHECKPOINT=checkpoint.json] [INITIAL_CHECKPOINT]
      Supervise operation generation on a JSON array of TrainingEpisode objects.
  collab-multi-context-llm run CHECKPOINT EPISODE [ROUNDS=256] [MEMORY_FILE]
      Free-run a learned model; EPISODE is one TrainingEpisode object.
      Demonstration rounds are ignored. Prints JSON trace to standard output.
      Optional memory file is loaded and saved after a completed task.

Local pretrained inference and distillation:
  collab-multi-context-llm infer --preset qwen|lfm -p PROMPT
  collab-multi-context-llm serve -m MODEL.gguf [--port 8080]
  collab-multi-context-llm distill --preset qwen|lfm --input PROMPTS.json --output SAMPLES.jsonl
  collab-multi-context-llm train-distill --input SAMPLES.jsonl --epochs N --output STUDENT.json
  collab-multi-context-llm student --checkpoint STUDENT.json -p PROMPT
  Use COMMAND --help for native setup and options.

The demo is scripted, not a pretrained agent. The bit experiment is an in-sample
smoke test, not evidence of general reasoning or learned operation generation.
";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("help" | "--help" | "-h") => {
            print!("{HELP}");
            Ok(())
        }
        Some("infer" | "serve" | "distill" | "train-distill" | "student") => {
            collab_multi_context_llm::commands::run(&args)
        }
        Some("demo") if args.len() <= 3 => demo(
            args.get(1).map_or("memory.json", String::as_str),
            args.get(2).map(String::as_str),
        ),
        Some("train") if args.len() <= 3 => {
            let steps = number(args.get(1), 12000, "steps")?;
            let checkpoint = args.get(2).map_or("checkpoint.json", String::as_str);
            let mut model = TinyTransformer::new(experiments::small_config(), 7)?;
            let (initial, final_loss) = experiments::train_published_bits(&mut model, steps)?;
            println!(
                "published-bit smoke test: {steps} SGD steps, mean loss {initial:.4} -> {final_loss:.4}"
            );
            println!(
                "in-sample stream accuracy: {}/8",
                experiments::published_bit_accuracy(&model, true)?
            );
            println!(
                "publication-disabled ablation: {}/8",
                experiments::published_bit_accuracy(&model, false)?
            );
            model.save(checkpoint)?;
            println!("saved {checkpoint}");
            Ok(())
        }
        Some("train-trace") if (3..=5).contains(&args.len()) => {
            let episodes: Vec<TrainingEpisode> = read_json(&args[1])?;
            if episodes.is_empty() {
                return Err("dataset must contain at least one episode".into());
            }
            let epochs = number(args.get(2), 1, "epochs")?;
            let mut config = experiments::small_config();
            config.max_streams = episodes
                .iter()
                .map(|e| e.working.len())
                .max()
                .unwrap()
                .checked_add(1)
                .ok_or("stream count overflow")?;
            let mut model = match args.get(4) {
                Some(initial) => TinyTransformer::load(initial)?,
                None => TinyTransformer::new(config, 7)?,
            };
            let budget = ContextBudget::default();
            // Validate the entire dataset before performing the first update.
            for episode in &episodes {
                training::validate_episode(episode, model.config(), &budget)?;
            }
            for epoch in 0..epochs {
                let mut total_loss = 0.0_f64;
                let mut tokens = 0;
                for episode in &episodes {
                    let stats = training::train_episode(&mut model, episode, 0.01, &budget)?;
                    total_loss += stats.mean_loss as f64 * stats.predicted_tokens as f64;
                    tokens += stats.predicted_tokens;
                }
                println!(
                    "epoch {}: token loss {:.4}, {tokens} supervised bytes",
                    epoch + 1,
                    total_loss / tokens as f64
                );
            }
            let checkpoint = args.get(3).map_or("checkpoint.json", String::as_str);
            model.save(checkpoint)?;
            println!("saved {checkpoint}; teacher-forced loss is not free-running task success");
            Ok(())
        }
        Some("run") if (3..=5).contains(&args.len()) => {
            let model = TinyTransformer::load(&args[1])?;
            let episode: TrainingEpisode = read_json(&args[2])?;
            let limit = number(args.get(3), 256, "rounds")?;
            let memory_path = args.get(4).map(String::as_str);
            let mut engine = match memory_path {
                Some(path) => training::initial_engine_with_memory(&episode, load_memory(path)?)?,
                None => training::initial_engine(&episode)?,
            };
            let budget = ContextBudget::default();
            for _ in 0..limit {
                if engine.is_finished() {
                    break;
                }
                let bytes = context::predict_round(&engine, &model, &budget)?;
                engine.step_tokens(bytes)?;
            }
            if engine.is_finished() {
                let committed = engine.commit_memory()?;
                if let Some(path) = memory_path {
                    engine.memory().save(path)?;
                    eprintln!(
                        "finished; {} memory entries committed, saved {path}",
                        committed.len()
                    );
                } else {
                    eprintln!(
                        "finished; {} memory entries committed in memory (not saved)",
                        committed.len()
                    );
                }
            } else {
                eprintln!("stopped at round limit {limit}; task unfinished");
            }
            serde_json::to_writer_pretty(std::io::stdout().lock(), engine.trace())
                .map_err(|e| e.to_string())?;
            println!();
            Ok(())
        }
        _ => Err(format!("invalid arguments\n\n{HELP}")),
    }
}

fn number(value: Option<&String>, default: usize, name: &str) -> Result<usize, String> {
    let value = match value {
        Some(value) => value
            .parse::<usize>()
            .map_err(|e| format!("invalid {name}: {e}"))?,
        None => default,
    };
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn load_memory(path: &str) -> Result<MemoryStore, String> {
    match Path::new(path).try_exists().map_err(|e| e.to_string())? {
        true => MemoryStore::load(path),
        false => Ok(MemoryStore::new(1024 * 1024)),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    serde_json::from_reader(BufReader::new(file)).map_err(|e| format!("read {path}: {e}"))
}

fn print_report(report: RoundReport) -> Result<(), String> {
    for outcome in report.outcomes {
        println!(
            "round {} stream {}: {}",
            report.round, outcome.stream, outcome.detail
        );
        if !outcome.accepted {
            return Err(format!("demo operation rejected: {}", outcome.detail));
        }
    }
    Ok(())
}

/// Deliver an entire action group one byte per stream per snapshot round, just
/// as the decoder would. This is scripted demonstration, not model inference.
fn deliver(engine: &mut Engine, actions: Vec<(StreamId, Operation)>) -> Result<(), String> {
    let frames: Vec<_> = actions
        .into_iter()
        .map(|(stream, operation)| (stream, operation.frame()))
        .collect();
    for position in 0..frames
        .iter()
        .map(|(_, frame)| frame.len())
        .max()
        .unwrap_or(0)
    {
        let bytes = frames
            .iter()
            .filter_map(|(stream, frame)| frame.get(position).map(|byte| (*stream, *byte)))
            .collect();
        print_report(engine.step_tokens(bytes)?)?;
    }
    Ok(())
}

fn demo(memory_path: &str, trace_path: Option<&str>) -> Result<(), String> {
    println!("scripted two-stream protocol demonstration (no learned policy)");
    let memory = load_memory(memory_path)?;
    let config = EngineConfig {
        scope: "demo".into(),
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(config.clone(), memory)?;
    engine.seed_entry(0, "private clue: a = 17")?;
    let entry = engine.seed_entry(1, "private clue: b = 25; hypothesis: total = 40")?;
    deliver(
        &mut engine,
        vec![
            (
                0,
                Operation::Publish {
                    target: None,
                    content: "a = 17".into(),
                },
            ),
            (
                1,
                Operation::Publish {
                    target: None,
                    content: "b = 25".into(),
                },
            ),
        ],
    )?;
    deliver(
        &mut engine,
        vec![(
            0,
            Operation::ProposeEdit {
                target: 1,
                entry_id: entry,
                expected_version: 1,
                replacement: "a = 17; b = 25; verified total = 42".into(),
            },
        )],
    )?;
    deliver(
        &mut engine,
        vec![(1, Operation::AcceptEdit { proposal_id: 1 })],
    )?;
    deliver(
        &mut engine,
        vec![(
            0,
            Operation::MemoryPropose {
                content: "Demo result: 17 + 25 = 42".into(),
                kind: MemoryKind::Fact,
                source_references: vec!["demo: arithmetic fixture 17 + 25".into()],
            },
        )],
    )?;
    deliver(
        &mut engine,
        vec![
            (
                0,
                Operation::Finish {
                    result: "42".into(),
                },
            ),
            (
                1,
                Operation::Finish {
                    result: "42".into(),
                },
            ),
        ],
    )?;
    let entries = engine.commit_memory()?;
    engine.memory().save(memory_path)?;
    println!(
        "committed {} memory entry; saved {memory_path}",
        entries.len()
    );
    if let Some(path) = trace_path {
        let mut writer =
            BufWriter::new(File::create(path).map_err(|e| format!("create {path}: {e}"))?);
        serde_json::to_writer_pretty(&mut writer, engine.trace()).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        println!("saved private diagnostic trace to {path}");
    }
    // Reopen storage into a fresh task to demonstrate real session persistence.
    let mut next_session = Engine::new(config, MemoryStore::load(memory_path)?)?;
    print_report(next_session.step_actions(vec![(
        1,
        Operation::MemoryQuery {
            query: "Demo result".into(),
        },
    )])?)?;
    for memory in next_session.streams()[&1].retrieved() {
        println!(
            "recalled memory {} v{}: {}",
            memory.id, memory.version, memory.content
        );
    }
    Ok(())
}
