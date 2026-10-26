//! In-process GGUF inference through the pinned llama.cpp C ABI.
//!
//! The external library supplies architecture implementations and CPU/GPU kernels;
//! it is not a subprocess or HTTP service. All llama parameter structs stay in C.

use std::ffi::{CStr, CString, c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::ptr::{self, NonNull};
use std::sync::OnceLock;

use super::{ChatMessage, Generation, GenerationConfig, InferenceBackend, validate_messages};

#[derive(Clone, Debug)]
pub struct NativeConfig {
    pub model: PathBuf,
    pub context_size: u32,
    pub batch_size: u32,
    pub threads: i32,
    pub gpu_layers: i32,
}

#[repr(C)]
struct Handle {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    fn ac_init(directory: *const c_char, error: *mut c_char, capacity: usize) -> c_int;
    fn ac_load(
        path: *const c_char,
        context_size: u32,
        batch_size: u32,
        threads: i32,
        gpu_layers: i32,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut Handle;
    fn ac_free(handle: *mut Handle);
    fn ac_reset(handle: *mut Handle);
    fn ac_context_size(handle: *const Handle) -> u32;
    fn ac_batch_size(handle: *const Handle) -> u32;
    fn ac_metadata(
        handle: *const Handle,
        key: *const c_char,
        output: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn ac_template_source(handle: *const Handle) -> *const c_char;
    fn ac_template(
        handle: *const Handle,
        roles: *const *const c_char,
        contents: *const *const c_char,
        count: usize,
        output: *mut c_char,
        capacity: i32,
    ) -> i32;
    fn ac_tokenize(
        handle: *const Handle,
        text: *const c_char,
        length: i32,
        tokens: *mut i32,
        capacity: i32,
    ) -> i32;
    fn ac_piece(handle: *const Handle, token: i32, output: *mut c_char, capacity: i32) -> i32;
    fn ac_begin(
        handle: *mut Handle,
        temperature: f32,
        top_p: f32,
        seed: u32,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn ac_decode(
        handle: *mut Handle,
        tokens: *const i32,
        count: i32,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn ac_sample(handle: *mut Handle) -> i32;
    fn ac_is_eog(handle: *const Handle, token: i32) -> c_int;
}

fn checked_call(call: impl FnOnce(*mut c_char, usize) -> c_int) -> Result<(), String> {
    let mut error = [0 as c_char; 1024];
    if call(error.as_mut_ptr(), error.len()) == 0 {
        Ok(())
    } else {
        // Every bridge error is NUL-terminated, including on truncation.
        Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }
}

fn initialize() -> Result<(), String> {
    static INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| {
            let directory = CString::new(env!("AGENT_LLAMA_CPP_LIB_DIR"))
                .map_err(|_| "native library directory contains NUL".to_owned())?;
            checked_call(|error, capacity| unsafe { ac_init(directory.as_ptr(), error, capacity) })
        })
        .clone()
}

// Exact generation branch inspected in Qwen3.5's embedded GGUF template. The
// pinned C API recognizes only its ChatML framing, not enable_thinking's branch.
const QWEN35_GENERATION_BRANCH: &str = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}";
const LFM_GENERATION_BRANCH: &str =
    "{%- if add_generation_prompt -%}\n    {{- \"<|im_start|>assistant\\n\" -}}\n{%- endif -%}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChatPolicy {
    Qwen35,
    Lfm2Moe,
    Other,
}

impl ChatPolicy {
    fn detect(architecture: &str, source: &str) -> Self {
        if architecture == "qwen35" && source.ends_with(QWEN35_GENERATION_BRANCH) {
            Self::Qwen35
        } else if architecture == "lfm2moe"
            && source.ends_with(LFM_GENERATION_BRANCH)
            && source.contains("preserve_thinking = preserve_thinking | default(false)")
            && source.contains("\"<think>\" + message.thinking + \"</think>\"")
        {
            Self::Lfm2Moe
        } else {
            Self::Other
        }
    }
}

/// Owns one model and one mutable inference context. Not shared between threads.
/// Drop frees the sampler, context, and model in that order. Global backend plugins
/// stay loaded so dropping a model cannot invalidate other live models.
pub struct NativeModel {
    handle: NonNull<Handle>,
    model_id: String,
    context_size: usize,
    batch_size: usize,
    thinking: bool,
    chat_policy: ChatPolicy,
}

impl Drop for NativeModel {
    fn drop(&mut self) {
        // The handle is uniquely owned and all bridge calls are synchronous.
        unsafe { ac_free(self.handle.as_ptr()) };
    }
}

impl NativeModel {
    pub fn load(config: NativeConfig) -> Result<Self, String> {
        if config.context_size == 0 || config.context_size > i32::MAX as u32 {
            return Err("context_size must be in 1..=i32::MAX".into());
        }
        if config.batch_size == 0 || config.batch_size > config.context_size {
            return Err("batch_size must be in 1..=context_size".into());
        }
        if config.threads <= 0 {
            return Err("threads must be positive".into());
        }
        if !config.model.is_file() {
            return Err(format!(
                "GGUF model is not a file: {}",
                config.model.display()
            ));
        }
        let path = CString::new(config.model.as_os_str().as_bytes())
            .map_err(|_| "model path contains NUL".to_owned())?;
        initialize()?;
        let mut raw = ptr::null_mut();
        checked_call(|error, capacity| {
            raw = unsafe {
                ac_load(
                    path.as_ptr(),
                    config.context_size,
                    config.batch_size,
                    config.threads,
                    config.gpu_layers,
                    error,
                    capacity,
                )
            };
            if raw.is_null() { -1 } else { 0 }
        })?;
        let handle = NonNull::new(raw).ok_or("native loader returned a null handle")?;
        let mut model = Self {
            handle,
            model_id: config.model.to_string_lossy().into_owned(),
            // llama.cpp may round context allocations; do not exceed the user's limit.
            context_size: unsafe { ac_context_size(raw) }.min(config.context_size) as usize,
            batch_size: unsafe { ac_batch_size(raw) }.min(config.batch_size) as usize,
            thinking: false,
            chat_policy: ChatPolicy::Other,
        };
        if model.context_size == 0 || model.batch_size == 0 {
            return Err("llama.cpp returned an empty context or batch capacity".into());
        }
        let architecture = model.metadata(c"general.architecture")?;
        let source = unsafe { ac_template_source(raw) };
        if !source.is_null() {
            let source = unsafe { CStr::from_ptr(source) }
                .to_str()
                .map_err(|_| "GGUF chat template is not UTF-8")?;
            model.chat_policy = ChatPolicy::detect(&architecture, source);
        }
        Ok(model)
    }

    fn metadata(&self, key: &CStr) -> Result<String, String> {
        let required =
            unsafe { ac_metadata(self.handle.as_ptr(), key.as_ptr(), ptr::null_mut(), 0) };
        if required < 0 {
            return Err(format!(
                "GGUF metadata is missing: {}",
                key.to_string_lossy()
            ));
        }
        let mut output = vec![0u8; required as usize + 1];
        let written = unsafe {
            ac_metadata(
                self.handle.as_ptr(),
                key.as_ptr(),
                output.as_mut_ptr().cast(),
                output.len(),
            )
        };
        if written != required {
            return Err("GGUF metadata changed size during retrieval".into());
        }
        output.truncate(written as usize);
        String::from_utf8(output).map_err(|_| "GGUF metadata is not UTF-8".into())
    }

    /// Default false. Qwen3.5 selects its embedded empty/open think branch.
    /// LFM2.5 has no verified off switch, so this setting cannot suppress its
    /// internal reasoning; unfinished reasoning is rejected instead.
    /// generate() always returns final text only, even when thinking is enabled.
    /// Raw complete() and apply_chat_template() are unaffected by this setting.
    pub fn set_thinking(&mut self, enabled: bool) {
        self.thinking = enabled;
    }

    fn generation_prompt(&self, messages: &[ChatMessage]) -> Result<(String, bool), String> {
        let mut prompt = self.apply_chat_template(messages)?;
        let mut prefilled_thinking = prompt.ends_with("<think>\n");
        if self.chat_policy != ChatPolicy::Other && !prompt.ends_with("<|im_start|>assistant\n") {
            return Err("native C template produced an unexpected assistant prefix".into());
        }
        if self.chat_policy == ChatPolicy::Qwen35 {
            // These are precisely the two suffixes in QWEN35_GENERATION_BRANCH,
            // not an invented template name or a replacement message format.
            if self.thinking {
                prompt.push_str("<think>\n");
                prefilled_thinking = true;
            } else {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        }
        Ok((prompt, prefilled_thinking))
    }

    /// Apply the GGUF's own supported template, including an assistant prefix.
    /// Unknown templates fail explicitly; there is no guessed ChatML fallback.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, String> {
        validate_messages(messages)?;
        let roles: Vec<CString> = messages
            .iter()
            .map(|message| CString::new(message.role.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|_| "chat role contains NUL".to_owned())?;
        let contents: Vec<CString> = messages
            .iter()
            .map(|message| CString::new(message.content.as_str()))
            .collect::<Result<_, _>>()
            .map_err(|_| "chat content contains NUL".to_owned())?;
        let role_ptrs: Vec<_> = roles.iter().map(|s| s.as_ptr()).collect();
        let content_ptrs: Vec<_> = contents.iter().map(|s| s.as_ptr()).collect();
        let apply = |output: *mut c_char, capacity| unsafe {
            ac_template(
                self.handle.as_ptr(),
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len(),
                output,
                capacity,
            )
        };
        let required = apply(ptr::null_mut(), 0);
        if required <= 0 {
            return Err("GGUF chat template is missing or unsupported by llama_chat_apply_template; use raw complete() instead".into());
        }
        let mut output = vec![0u8; required as usize];
        let written = apply(output.as_mut_ptr().cast(), required);
        if written < 0 || written > required {
            return Err("llama_chat_apply_template failed or changed output size".into());
        }
        output.truncate(written as usize);
        String::from_utf8(output)
            .map_err(|error| format!("chat template returned invalid UTF-8: {error}"))
    }

    /// Raw completion: no chat template or assistant prefix is added. The model's
    /// tokenizer still inserts its configured BOS/EOS and recognizes special tokens.
    pub fn complete(
        &mut self,
        prompt: &str,
        options: &GenerationConfig,
    ) -> Result<Generation, String> {
        options.validate()?;
        let result = self.complete_inner(prompt, options);
        // Also clear recurrent memory after decode failures and early stop returns.
        unsafe { ac_reset(self.handle.as_ptr()) };
        result
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<i32>, String> {
        let length = i32::try_from(prompt.len()).map_err(|_| "prompt exceeds i32::MAX bytes")?;
        let count = unsafe {
            ac_tokenize(
                self.handle.as_ptr(),
                prompt.as_ptr().cast(),
                length,
                ptr::null_mut(),
                0,
            )
        };
        let count = count
            .checked_neg()
            .filter(|count| *count > 0)
            .ok_or("tokenizer returned no tokens or overflowed")?;
        if count as usize > self.context_size {
            return Err(format!(
                "prompt has {count} tokens but context capacity is {}",
                self.context_size
            ));
        }
        let mut tokens = vec![0; count as usize];
        let written = unsafe {
            ac_tokenize(
                self.handle.as_ptr(),
                prompt.as_ptr().cast(),
                length,
                tokens.as_mut_ptr(),
                count,
            )
        };
        if written <= 0 || written > count {
            return Err("llama_tokenize failed or changed token count".into());
        }
        tokens.truncate(written as usize);
        Ok(tokens)
    }

    fn decode(&mut self, tokens: &[i32]) -> Result<(), String> {
        let count = i32::try_from(tokens.len()).map_err(|_| "decode batch exceeds i32::MAX")?;
        checked_call(|error, capacity| unsafe {
            ac_decode(
                self.handle.as_ptr(),
                tokens.as_ptr(),
                count,
                error,
                capacity,
            )
        })
    }

    fn piece(&self, token: i32) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; 128];
        let mut written =
            unsafe { ac_piece(self.handle.as_ptr(), token, bytes.as_mut_ptr().cast(), 128) };
        if written < 0 {
            let required = written.checked_neg().ok_or("token piece size overflow")?;
            bytes.resize(required as usize, 0);
            written = unsafe {
                ac_piece(
                    self.handle.as_ptr(),
                    token,
                    bytes.as_mut_ptr().cast(),
                    required,
                )
            };
        }
        if written < 0 || written as usize > bytes.len() {
            return Err("llama_token_to_piece failed".into());
        }
        bytes.truncate(written as usize);
        Ok(bytes)
    }

    fn complete_inner(
        &mut self,
        prompt: &str,
        options: &GenerationConfig,
    ) -> Result<Generation, String> {
        let tokens = self.tokenize(prompt)?;
        if options.max_tokens > self.context_size - tokens.len() {
            return Err(format!(
                "prompt ({} tokens) plus max_tokens ({}) exceeds context capacity ({})",
                tokens.len(),
                options.max_tokens,
                self.context_size
            ));
        }
        checked_call(|error, capacity| unsafe {
            ac_begin(
                self.handle.as_ptr(),
                options.temperature,
                options.top_p,
                options.seed,
                error,
                capacity,
            )
        })?;
        for batch in tokens.chunks(self.batch_size) {
            self.decode(batch)?;
        }
        let mut bytes = Vec::new();
        let mut completion_tokens = 0;
        let mut finish_reason = "length";
        for index in 0..options.max_tokens {
            let token = unsafe { ac_sample(self.handle.as_ptr()) };
            if token < 0 {
                return Err("llama_sampler_sample returned an invalid token".into());
            }
            completion_tokens += 1;
            if unsafe { ac_is_eog(self.handle.as_ptr(), token) } != 0 {
                finish_reason = "stop";
                break;
            }
            bytes.extend_from_slice(&self.piece(token)?);
            if let Some(position) = stop_position(&bytes, &options.stop) {
                bytes.truncate(position);
                finish_reason = "stop";
                break;
            }
            // The final sampled token need not be decoded to obtain more logits.
            if index + 1 < options.max_tokens {
                self.decode(&[token])?;
            }
        }
        Ok(Generation {
            // Token pieces can divide a multibyte codepoint. Convert exactly once,
            // replacing only invalid bytes or an incomplete final codepoint.
            text: String::from_utf8_lossy(&bytes).into_owned(),
            model: self.model_id.clone(),
            prompt_tokens: tokens.len(),
            completion_tokens,
            finish_reason: finish_reason.into(),
        })
    }
}

impl InferenceBackend for NativeModel {
    fn complete_raw(
        &mut self,
        prompt: &str,
        options: &GenerationConfig,
    ) -> Result<Generation, String> {
        self.complete(prompt, options)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn generate(
        &mut self,
        messages: &[ChatMessage],
        options: &GenerationConfig,
    ) -> Result<Generation, String> {
        options.validate()?;
        let (prompt, prefilled_thinking) = self.generation_prompt(messages)?;
        let mut generation = self.complete(&prompt, options)?;
        generation.text =
            final_text(&generation.text, prefilled_thinking, self.chat_policy)?.to_owned();
        Ok(generation)
    }
}

fn final_text(text: &str, prefilled_thinking: bool, policy: ChatPolicy) -> Result<&str, String> {
    let leading = text.trim_start();
    let reasoning = if prefilled_thinking {
        Some(text)
    } else {
        leading.strip_prefix("<think>")
    };
    if let Some(reasoning) = reasoning {
        if policy == ChatPolicy::Other {
            return Err("native reasoning output has an unrecognized template; refusing to expose hidden thoughts".into());
        }
        let (_, answer) = reasoning.split_once("</think>").ok_or(
            "native reasoning block is unfinished; increase max_tokens (no final answer returned)",
        )?;
        let answer = answer.trim_start();
        if answer.trim().is_empty() {
            return Err("native reasoning ended without a final answer".into());
        }
        if answer.starts_with("<think>")
            || "<think>".starts_with(answer)
            || answer.starts_with("</think>")
        {
            return Err("native output has repeated or ambiguous leading reasoning blocks".into());
        }
        // Remove exactly one leading protocol block, never tags inside the answer.
        return Ok(answer);
    }
    if (!leading.is_empty() && "<think>".starts_with(leading)) || leading.starts_with("</think>") {
        return Err(
            "native output contains an incomplete or unexpected reasoning delimiter".into(),
        );
    }
    Ok(text)
}

fn stop_position(bytes: &[u8], stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| {
            bytes
                .windows(stop.len())
                .position(|window| window == stop.as_bytes())
        })
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_matching_is_byte_based_and_earliest() {
        let mut bytes = vec![b'A', 0xe2];
        bytes.extend_from_slice(&[0x82, 0xac, b'B']);
        assert_eq!(String::from_utf8_lossy(&bytes), "A€B");
        assert_eq!(stop_position(&bytes, &["B".into(), "€".into()]), Some(1));
        assert_eq!(stop_position(b"abST", &["STOP".into()]), None);
        assert_eq!(stop_position(b"abSTOPcd", &["STOP".into()]), Some(2));
        assert_eq!(stop_position(b"a", &["longer".into()]), None);
    }

    #[test]
    fn reasoning_policy_requires_architecture_and_verified_source() {
        assert_eq!(
            ChatPolicy::detect("qwen35", QWEN35_GENERATION_BRANCH),
            ChatPolicy::Qwen35
        );
        assert_eq!(
            ChatPolicy::detect("other", QWEN35_GENERATION_BRANCH),
            ChatPolicy::Other
        );
        assert_eq!(ChatPolicy::detect("qwen35", "chatml"), ChatPolicy::Other);
        let lfm = format!(
            "preserve_thinking = preserve_thinking | default(false)\n\"<think>\" + message.thinking + \"</think>\"\n{LFM_GENERATION_BRANCH}"
        );
        assert_eq!(ChatPolicy::detect("lfm2moe", &lfm), ChatPolicy::Lfm2Moe);
        assert_eq!(ChatPolicy::detect("qwen35", &lfm), ChatPolicy::Other);
    }

    #[test]
    fn only_one_completed_leading_reasoning_block_is_removed() {
        let policy = ChatPolicy::Lfm2Moe;
        assert_eq!(
            final_text(" \n<think>private</think>\n\n4", false, policy).unwrap(),
            "4"
        );
        assert_eq!(
            final_text("private</think>\n4", true, ChatPolicy::Qwen35).unwrap(),
            "4"
        );
        assert!(final_text("<think>unfinished", false, policy).is_err());
        assert!(final_text("unfinished", true, ChatPolicy::Qwen35).is_err());
        assert!(final_text("<think>done</think>\n", false, policy).is_err());
        assert!(final_text("<think>private</think>4", false, ChatPolicy::Other).is_err());
        assert!(final_text("<thi", false, policy).is_err());
        assert!(final_text("</think>4", false, policy).is_err());
        assert!(final_text("<think>one</think><think>two</think>4", false, policy).is_err());
        let literal = "The literal <think> tag appears here.";
        assert_eq!(final_text(literal, false, policy).unwrap(), literal);
        let once = "<think>private</think>Answer quotes <think>literal</think>";
        assert_eq!(
            final_text(once, false, policy).unwrap(),
            "Answer quotes <think>literal</think>"
        );
    }

    #[test]
    fn invalid_configuration_fails_before_loading() {
        let result = NativeModel::load(NativeConfig {
            model: PathBuf::from("not-a-model.gguf"),
            context_size: 0,
            batch_size: 32,
            threads: 2,
            gpu_layers: 0,
        });
        assert!(result.err().unwrap().contains("context_size"));
    }

    fn live_model(variable: &str) {
        let model = std::env::var_os(variable)
            .unwrap_or_else(|| panic!("set {variable} to the actual GGUF path"));
        let mut model = NativeModel::load(NativeConfig {
            model: model.into(),
            context_size: 512,
            batch_size: 32,
            threads: 2,
            gpu_layers: 0,
        })
        .unwrap();
        let options = GenerationConfig {
            max_tokens: 8,
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        let messages = [
            ChatMessage::system("You are a helpful assistant. ".repeat(8)),
            ChatMessage::user("Reply with just the word hello."),
        ];
        let template = model.apply_chat_template(&messages).unwrap();
        assert!(template.contains(&messages[0].content));
        assert!(template.contains(&messages[1].content));
        assert!(template.contains("assistant"));
        eprintln!("{} template: {template:?}", model.model_id());
        // Preserve the original eight-token raw-template test. generate() must
        // now reject unfinished reasoning rather than return it as final text.
        let chat = model.complete(&template, &options).unwrap();
        assert!(chat.prompt_tokens > model.batch_size);
        assert!(chat.completion_tokens > 0 && chat.completion_tokens <= 8);
        assert!(!chat.text.is_empty());
        eprintln!("raw-template chat: {chat:?}");
        assert!(matches!(
            model.chat_policy,
            ChatPolicy::Qwen35 | ChatPolicy::Lfm2Moe
        ));
        let arithmetic = [ChatMessage::user(
            "What is 2 + 2? Reply with only the digit.",
        )];
        let final_options = GenerationConfig {
            max_tokens: 64,
            ..options.clone()
        };
        let (final_prompt, prefilled) = model.generation_prompt(&arithmetic).unwrap();
        assert!(!prefilled);
        eprintln!("final prompt: {final_prompt:?}");
        let answer = match model.generate(&arithmetic, &final_options) {
            Ok(answer) => answer,
            Err(error) if model.chat_policy == ChatPolicy::Lfm2Moe => {
                assert!(error.contains("unfinished"));
                eprintln!("64-token LFM final-answer request safely rejected: {error}");
                model
                    .generate(
                        &arithmetic,
                        &GenerationConfig {
                            max_tokens: 128,
                            ..final_options.clone()
                        },
                    )
                    .unwrap()
            }
            Err(error) => panic!("final generation failed: {error}"),
        };
        eprintln!("final answer: {answer:?}");
        assert_eq!(answer.text.trim(), "4");
        assert_eq!(answer.finish_reason, "stop");
        assert!(
            answer.completion_tokens
                <= if model.chat_policy == ChatPolicy::Qwen35 {
                    64
                } else {
                    128
                }
        );
        model.set_thinking(true);
        assert_eq!(model.apply_chat_template(&messages).unwrap(), template);
        let (thinking_prompt, prefilled) = model.generation_prompt(&arithmetic).unwrap();
        if model.chat_policy == ChatPolicy::Qwen35 {
            assert!(prefilled);
            assert!(thinking_prompt.ends_with("<think>\n"));
        }
        let unfinished = model.generate(&arithmetic, &options);
        assert!(unfinished.unwrap_err().contains("unfinished"));
        model.set_thinking(false);

        // Raw completion, small prefill batches, and hybrid memory reset after chat.
        let prompt = "The capital of France is";
        let raw = model.complete(prompt, &options).unwrap();
        assert!(!raw.text.is_empty());
        assert!(raw.completion_tokens > 0 && raw.completion_tokens <= 8);
        let repeat = model.complete(prompt, &options).unwrap();
        assert_eq!(raw.text, repeat.text);
        assert_eq!(raw.completion_tokens, repeat.completion_tokens);
        eprintln!("raw: {raw:?}");

        let stop = raw.text.chars().next().unwrap().to_string();
        let stopped = model
            .complete(
                prompt,
                &GenerationConfig {
                    stop: vec![stop],
                    ..options.clone()
                },
            )
            .unwrap();
        assert_eq!(stopped.finish_reason, "stop");
        assert!(stopped.text.is_empty());
        let sampled_options = GenerationConfig {
            temperature: 0.6,
            seed: 123,
            ..options.clone()
        };
        let sampled = model.complete(prompt, &sampled_options).unwrap();
        let repeated = model.complete(prompt, &sampled_options).unwrap();
        assert_eq!(sampled.text, repeated.text);
        assert_eq!(sampled.completion_tokens, repeated.completion_tokens);
        assert!(model.complete(&"hello ".repeat(600), &options).is_err());
        assert!(
            model
                .complete(
                    prompt,
                    &GenerationConfig {
                        max_tokens: 512,
                        ..options.clone()
                    }
                )
                .is_err()
        );
        assert!(
            model
                .complete(
                    prompt,
                    &GenerationConfig {
                        max_tokens: 0,
                        ..options
                    }
                )
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires NATIVE_TEST_QWEN_MODEL and the compatible llama.cpp shared libraries"]
    fn live_qwen_gguf() {
        live_model("NATIVE_TEST_QWEN_MODEL");
    }

    #[test]
    #[ignore = "requires NATIVE_TEST_LFM_MODEL and the compatible llama.cpp shared libraries"]
    fn live_lfm_gguf() {
        live_model("NATIVE_TEST_LFM_MODEL");
    }
}
