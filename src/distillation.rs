//! Sequence-level distillation from a teacher's final answer text, not its logits.
//!
//! Teacher token counts are provenance only: the student learns UTF-8 bytes with
//! its own vocabulary. This small model does not inherit a 4B teacher's general
//! capabilities. No reasoning-content field is consumed or used as a fallback.
//! Responses starting with `<think>` after whitespace, or containing `</think>`
//! anywhere, are rejected rather than extracting a guessed final answer. This
//! conservative, case-sensitive check permits quoted opening tags inside ordinary
//! text, but rejects even quoted closing tags because their meaning is ambiguous.
//! It detects these known markers, not arbitrary untagged reasoning.
//!
//! Inputs are BOS followed by the most recent `max_positions - 1` bytes of the
//! role-labelled prompt and answer prefix. Older context (including the prompt)
//! can slide out, but every answer byte is supervised in every epoch. BOS is
//! input-only: no EOS, newline, or other synthetic target is appended.

use crate::inference::{ChatMessage, GenerationConfig, InferenceBackend, validate_messages};
use crate::model::{BOS, InputToken, ModelConfig, ModelInput, TinyTransformer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FORMAT_VERSION: u32 = 1;
/// Maximum encoded record size, including its terminating newline if present.
pub const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillPrompt {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub expected_answer: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherSample {
    pub format_version: u32,
    pub id: String,
    pub messages: Vec<ChatMessage>,
    /// Actual response model ID; it may differ from the requested backend alias.
    pub teacher: String,
    pub response: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub generation: GenerationConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillStats {
    /// Byte-weighted mean cross entropy before any updates (natural logarithm).
    pub initial_loss: f32,
    /// Byte-weighted mean cross entropy after all updates, on the same samples.
    pub final_loss: f32,
    /// Number of supervised response bytes, including repetitions across epochs.
    /// This does not count prompt bytes or report retained context length.
    pub trained_bytes: usize,
}

fn nonempty(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} cannot be blank"))
    } else {
        Ok(())
    }
}

fn validate_sample(sample: &TeacherSample) -> Result<(), String> {
    if sample.format_version != FORMAT_VERSION {
        return Err(format!(
            "unsupported sample format version {}",
            sample.format_version
        ));
    }
    nonempty(&sample.id, "sample ID")?;
    validate_messages(&sample.messages)?;
    nonempty(&sample.teacher, "teacher model ID")?;
    nonempty(&sample.response, "teacher response")?;
    if sample.response.trim_start().starts_with("<think>") || sample.response.contains("</think>") {
        return Err("teacher response contains raw reasoning markers; a final answer without think blocks is required".into());
    }
    nonempty(&sample.finish_reason, "finish reason")?;
    if sample.finish_reason.trim().eq_ignore_ascii_case("length") {
        return Err("teacher response was truncated (finish_reason=length)".into());
    }
    sample.generation.validate()
}

fn validate_samples(samples: &[TeacherSample]) -> Result<usize, String> {
    if samples.is_empty() {
        return Err("at least one teacher sample is required".into());
    }
    let mut identities = HashSet::new();
    let mut bytes = 0usize;
    for (index, sample) in samples.iter().enumerate() {
        validate_sample(sample).map_err(|e| format!("sample {index} ({:?}): {e}", sample.id))?;
        if !identities.insert((&sample.id, &sample.teacher)) {
            return Err(format!(
                "duplicate sample ID {:?} for teacher {:?}",
                sample.id, sample.teacher
            ));
        }
        bytes = bytes
            .checked_add(sample.response.len())
            .ok_or("response byte count overflow")?;
    }
    Ok(bytes)
}

/// Collect complete final answers. All prompts and options are validated before
/// contacting the backend. An expected answer is compared exactly after trimming
/// both ends; the original teacher response is retained without modification.
/// A backend alias need not equal the actual response model ID.
pub fn collect(
    backend: &mut dyn InferenceBackend,
    prompts: &[DistillPrompt],
    options: &GenerationConfig,
) -> Result<Vec<TeacherSample>, String> {
    options.validate()?;
    if prompts.is_empty() {
        return Err("at least one distillation prompt is required".into());
    }
    let mut ids = HashSet::new();
    for prompt in prompts {
        nonempty(&prompt.id, "prompt ID")?;
        if !ids.insert(&prompt.id) {
            return Err(format!("duplicate prompt ID {:?}", prompt.id));
        }
        validate_messages(&prompt.messages).map_err(|e| format!("prompt {:?}: {e}", prompt.id))?;
        if let Some(expected) = &prompt.expected_answer {
            nonempty(expected, "expected answer")?;
        }
    }
    let mut samples = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let answer = backend
            .generate(&prompt.messages, options)
            .map_err(|e| format!("prompt {:?}: teacher generation failed: {e}", prompt.id))?;
        let sample = TeacherSample {
            format_version: FORMAT_VERSION,
            id: prompt.id.clone(),
            messages: prompt.messages.clone(),
            teacher: answer.model,
            response: answer.text,
            prompt_tokens: answer.prompt_tokens,
            completion_tokens: answer.completion_tokens,
            finish_reason: answer.finish_reason,
            generation: options.clone(),
        };
        validate_sample(&sample).map_err(|e| format!("prompt {:?}: {e}", prompt.id))?;
        if let Some(expected) = &prompt.expected_answer
            && sample.response.trim() != expected.trim()
        {
            return Err(format!(
                "prompt {:?}: teacher response does not match expected answer",
                prompt.id
            ));
        }
        samples.push(sample);
    }
    Ok(samples)
}

/// Remove an unpublished temporary file on every error path.
struct TemporaryPath(PathBuf);
impl Drop for TemporaryPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn temporary_file(destination: &Path) -> Result<(TemporaryPath, File), String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if destination.file_name().is_none() {
        return Err("dataset path must name a file".into());
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    for _ in 0..128 {
        let path = parent.join(format!(
            ".distillation-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((TemporaryPath(path), file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!(
                    "create dataset temporary file in {}: {e}",
                    parent.display()
                ));
            }
        }
    }
    Err("could not allocate a unique dataset temporary file".into())
}

/// Bound encoded record size during serialization, without first allocating an
/// arbitrarily large JSON buffer (escaping can expand the source strings).
struct RecordWriter<W> {
    inner: W,
    remaining: usize,
}

impl<W: Write> Write for RecordWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSONL record exceeds {MAX_LINE_BYTES} bytes including newline"),
            ));
        }
        let written = self.inner.write(bytes)?;
        self.remaining -= written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write a new JSONL dataset without overwriting any existing destination.
/// The complete file is flushed and synced before atomic no-clobber publication
/// using a same-directory hard link (rename would overwrite on Unix). On failure
/// the destination is untouched and the temporary file is cleaned up. Requires
/// a filesystem supporting hard links; no unsafe overwrite fallback is used.
pub fn write_jsonl(path: impl AsRef<Path>, samples: &[TeacherSample]) -> Result<(), String> {
    validate_samples(samples)?;
    let path = path.as_ref();
    let (temporary, file) = temporary_file(path)?;
    let mut writer = BufWriter::new(file);
    for sample in samples {
        serde_json::to_writer(
            &mut RecordWriter {
                inner: &mut writer,
                remaining: MAX_LINE_BYTES - 1,
            },
            sample,
        )
        .map_err(|e| {
            format!(
                "serialize/write sample {:?} in {}: {e}",
                sample.id,
                path.display()
            )
        })?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("write dataset {}: {e}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush dataset {}: {e}", path.display()))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| format!("sync dataset {}: {e}", path.display()))?;
    drop(writer);
    fs::hard_link(&temporary.0, path).map_err(|e| {
        format!(
            "publish new dataset {} (destination must not exist): {e}",
            path.display()
        )
    })?;
    Ok(())
}

/// Read strict versioned JSONL with bounded allocation per line. Blank lines,
/// unknown fields, malformed records, and duplicate (ID, teacher) pairs fail the
/// whole read. The same ID from different teachers is allowed when concatenating
/// datasets. Invalid records are rejected before reading subsequent lines. A final
/// newline is optional.
pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<TeacherSample>, String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| format!("open dataset {}: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut samples = Vec::new();
    let mut identities = HashSet::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let number = samples.len() + 1;
        let count = (&mut reader)
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("read dataset {}, line {number}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        if count > MAX_LINE_BYTES {
            return Err(format!(
                "dataset {}, line {number}: exceeds {MAX_LINE_BYTES} bytes",
                path.display()
            ));
        }
        let sample = serde_json::from_slice::<TeacherSample>(&line)
            .map_err(|e| format!("dataset {}, line {number}: {e}", path.display()))?;
        validate_sample(&sample)
            .map_err(|e| format!("dataset {}, line {number}: {e}", path.display()))?;
        if !identities.insert((sample.id.clone(), sample.teacher.clone())) {
            return Err(format!(
                "dataset {}, line {number}: duplicate sample ID {:?} for teacher {:?}",
                path.display(),
                sample.id,
                sample.teacher
            ));
        }
        samples.push(sample);
    }
    if samples.is_empty() {
        return Err(format!(
            "dataset {}: at least one teacher sample is required",
            path.display()
        ));
    }
    Ok(samples)
}

fn validate_model(config: &ModelConfig) -> Result<(), String> {
    // The final identity is reserved for neutral/shared context, never training.
    if config.max_streams < 2 || config.max_positions < 2 || config.context_kinds == 0 {
        return Err("distillation requires at least two stream identities (one neutral), two positions, and one context kind".into());
    }
    Ok(())
}

fn prompt_bytes(messages: &[ChatMessage]) -> Vec<u8> {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(&message.role);
        prompt.push_str(":\n");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt.push_str("assistant:\n");
    prompt.into_bytes()
}

/// Shared training/inference encoding. The target byte is never in the input.
fn answer_input(config: &ModelConfig, prompt: &[u8], prefix: &[u8], stream: usize) -> ModelInput {
    let capacity = config.max_positions - 1;
    let history_len = prefix.len().min(capacity);
    let prompt_len = prompt.len().min(capacity - history_len);
    let mut tokens = Vec::with_capacity(1 + prompt_len + history_len);
    tokens.push(InputToken {
        token: BOS,
        position: 0,
        stream,
        kind: 0,
    });
    for &byte in prompt[prompt.len() - prompt_len..]
        .iter()
        .chain(&prefix[prefix.len() - history_len..])
    {
        tokens.push(InputToken {
            token: usize::from(byte),
            position: tokens.len(),
            stream,
            kind: 0,
        });
    }
    let n = tokens.len();
    let visible = (0..n)
        .map(|row| (0..n).map(|col| col <= row).collect())
        .collect();
    ModelInput {
        tokens,
        visible,
        queries: vec![n - 1],
    }
}

fn visit_targets(
    config: &ModelConfig,
    samples: &[TeacherSample],
    mut visit: impl FnMut(&ModelInput, usize) -> Result<(), String>,
) -> Result<(), String> {
    for (index, sample) in samples.iter().enumerate() {
        let prompt = prompt_bytes(&sample.messages);
        let response = sample.response.as_bytes();
        let stream = index % (config.max_streams - 1);
        for (offset, &byte) in response.iter().enumerate() {
            let input = answer_input(config, &prompt, &response[..offset], stream);
            visit(&input, usize::from(byte))
                .map_err(|e| format!("sample {:?}, answer byte {offset}: {e}", sample.id))?;
        }
    }
    Ok(())
}

fn mean_loss(
    model: &TinyTransformer,
    samples: &[TeacherSample],
    bytes: usize,
) -> Result<f32, String> {
    let mut total = 0.0f64;
    visit_targets(model.config(), samples, |input, target| {
        let predictions = model.predict(input)?;
        // predict exposes f32 probabilities, not logits: floor only numerical
        // underflow for evaluation. SGD uses the model's stable logit-space CE.
        total -= f64::from(predictions[0][target].max(f32::MIN_POSITIVE)).ln();
        Ok(())
    })?;
    Ok((total / bytes as f64) as f32)
}

/// Train only next-answer-byte loss, never prompt loss. Each epoch visits every
/// response byte, including UTF-8 continuation bytes and long-answer tails.
/// Sample indices rotate through all non-neutral stream identities.
///
/// Dataset, hyperparameter, byte-count, and model validation precede updates.
/// Numerical failures after earlier successful steps are not rolled back.
/// Losses are evaluated on the training samples, not a held-out quality metric.
pub fn train_samples(
    model: &mut TinyTransformer,
    samples: &[TeacherSample],
    epochs: usize,
    learning_rate: f32,
) -> Result<DistillStats, String> {
    if epochs == 0 {
        return Err("epochs must be positive".into());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive".into());
    }
    let bytes = validate_samples(samples)?;
    let trained_bytes = bytes
        .checked_mul(epochs)
        .ok_or("trained byte count overflow")?;
    let config = model.config().clone();
    validate_model(&config)?;
    let initial_loss = mean_loss(model, samples, bytes)?;
    for _ in 0..epochs {
        visit_targets(&config, samples, |input, target| {
            model
                .train_step(input, &[target], learning_rate)
                .map(|_| ())
        })?;
    }
    let final_loss = mean_loss(model, samples, bytes)?;
    Ok(DistillStats {
        initial_loss,
        final_loss,
        trained_bytes,
    })
}

/// Greedily generate exactly `max_bytes` bytes with the training prompt format,
/// using stream identity zero and never emitting BOS. There is no learned EOS or
/// finish detection: the byte cap is the only stopping rule. Returns an error if
/// the generated bytes are invalid UTF-8 (including a cap splitting a codepoint).
pub fn predict_answer(
    model: &TinyTransformer,
    messages: &[ChatMessage],
    max_bytes: usize,
) -> Result<String, String> {
    validate_messages(messages)?;
    validate_model(model.config())?;
    if max_bytes == 0 {
        return Err("max_bytes must be positive".into());
    }
    let prompt = prompt_bytes(messages);
    let mut answer = Vec::new();
    for _ in 0..max_bytes {
        let input = answer_input(model.config(), &prompt, &answer, 0);
        let predictions = model.predict(&input)?;
        let byte = (0..256)
            .max_by(|&a, &b| predictions[0][a].total_cmp(&predictions[0][b]))
            .ok_or("no byte predictions")?;
        answer.push(byte as u8);
    }
    String::from_utf8(answer).map_err(|e| format!("student generated invalid UTF-8: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::Generation;

    struct FakeTeacher {
        calls: usize,
        answer: Generation,
    }

    impl InferenceBackend for FakeTeacher {
        fn model_id(&self) -> &str {
            "local-alias"
        }
        fn generate(
            &mut self,
            messages: &[ChatMessage],
            options: &GenerationConfig,
        ) -> Result<Generation, String> {
            validate_messages(messages)?;
            options.validate()?;
            self.calls += 1;
            Ok(self.answer.clone())
        }
    }

    fn teacher() -> FakeTeacher {
        FakeTeacher {
            calls: 0,
            answer: Generation {
                text: "  yes\n".into(),
                model: "actual-qwen-model".into(),
                prompt_tokens: 7,
                completion_tokens: 2,
                finish_reason: "stop".into(),
            },
        }
    }

    fn prompt(id: &str) -> DistillPrompt {
        DistillPrompt {
            id: id.into(),
            messages: vec![ChatMessage::user("Say yes")],
            expected_answer: Some("yes".into()),
        }
    }

    fn sample() -> TeacherSample {
        collect(
            &mut teacher(),
            &[prompt("p1")],
            &GenerationConfig::default(),
        )
        .unwrap()
        .remove(0)
    }

    fn model(positions: usize) -> TinyTransformer {
        TinyTransformer::new(
            ModelConfig {
                width: 8,
                max_positions: positions,
                max_streams: 3,
                context_kinds: 1,
            },
            42,
        )
        .unwrap()
    }

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "distillation-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> PathBuf {
            self.0.join("samples.jsonl")
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn collects_final_text_with_actual_model_provenance_and_trimmed_exact_check() {
        let mut backend = teacher();
        let options = GenerationConfig {
            seed: 19,
            ..GenerationConfig::default()
        };
        let samples = collect(&mut backend, &[prompt("p1"), prompt("p2")], &options).unwrap();
        assert_eq!(backend.calls, 2);
        assert_eq!(samples[0].teacher, "actual-qwen-model");
        assert_ne!(samples[0].teacher, backend.model_id());
        assert_eq!(samples[0].response, "  yes\n");
        assert_eq!(samples[0].generation.seed, 19);
        assert_eq!(
            (samples[0].prompt_tokens, samples[0].completion_tokens),
            (7, 2)
        );
        let mut unchecked = prompt("unchecked");
        unchecked.expected_answer = None;
        backend.answer.text = "a different final answer".into();
        assert!(collect(&mut backend, &[unchecked], &options).is_ok());
    }

    #[test]
    fn collection_rejects_blank_truncated_and_nonmatching_answers() {
        for (text, reason) in [
            (" ", "stop"),
            ("yes", "length"),
            ("yes", " LENGTH "),
            ("Yes", "stop"),
            ("yes and more", "stop"),
        ] {
            let mut backend = teacher();
            backend.answer.text = text.into();
            backend.answer.finish_reason = reason.into();
            assert!(collect(&mut backend, &[prompt("p")], &GenerationConfig::default()).is_err());
        }
        let mut backend = teacher();
        backend.answer.model.clear();
        assert!(collect(&mut backend, &[prompt("p")], &GenerationConfig::default()).is_err());
    }

    #[test]
    fn collection_rejects_completed_reasoning_blocks() {
        for response in [
            "<think>private reasoning</think>yes",
            " \n\t<think>private reasoning</think>\nyes",
        ] {
            let mut backend = teacher();
            backend.answer.text = response.into();
            let mut prompt = prompt("completed");
            // Even an exact expected-answer match must not bypass validation.
            prompt.expected_answer = Some(response.into());
            let error = collect(&mut backend, &[prompt], &GenerationConfig::default()).unwrap_err();
            assert!(error.contains("raw reasoning markers"), "{error}");
        }
    }

    #[test]
    fn collection_rejects_unclosed_reasoning_blocks() {
        for response in [
            "<think>",
            "<think>unfinished reasoning",
            "\u{2003}\n<think>unfinished reasoning",
        ] {
            let mut backend = teacher();
            backend.answer.text = response.into();
            let mut prompt = prompt("unclosed");
            prompt.expected_answer = None;
            let error = collect(&mut backend, &[prompt], &GenerationConfig::default()).unwrap_err();
            assert!(error.contains("raw reasoning markers"), "{error}");
        }
    }

    #[test]
    fn collection_rejects_stray_closing_think_tags() {
        for response in [
            "</think>yes",
            "reasoning tail</think>yes",
            "The closing tag is '</think>'.",
        ] {
            let mut backend = teacher();
            backend.answer.text = response.into();
            let mut prompt = prompt("stray-close");
            prompt.expected_answer = None;
            let error = collect(&mut backend, &[prompt], &GenerationConfig::default()).unwrap_err();
            assert!(error.contains("raw reasoning markers"), "{error}");
        }
    }

    #[test]
    fn reasoning_contamination_is_rejected_across_the_data_lifecycle() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let mut model = model(8);
        let input = answer_input(model.config(), b"prompt", &[], 0);
        let before = model.predict(&input).unwrap();
        let valid = sample();
        for response in [
            "<think>reasoning</think>yes",
            " \n<think>unfinished",
            "reasoning tail</think>yes",
        ] {
            let mut contaminated = valid.clone();
            contaminated.id = "contaminated".into();
            contaminated.response = response.into();
            let samples = [valid.clone(), contaminated];
            let error = write_jsonl(&path, &samples).unwrap_err();
            assert!(error.contains("raw reasoning markers"), "{error}");
            assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
            let data = samples
                .iter()
                .map(|s| serde_json::to_string(s).unwrap() + "\n")
                .collect::<String>();
            fs::write(&path, &data).unwrap();
            let error = read_jsonl(&path).unwrap_err();
            assert!(
                error.contains("line 2") && error.contains("raw reasoning markers"),
                "{error}"
            );
            let error = train_samples(&mut model, &samples, 1, 0.01).unwrap_err();
            assert!(error.contains("raw reasoning markers"), "{error}");
            assert_eq!(model.predict(&input).unwrap(), before);
            // Rejected data must not replace an existing dataset either.
            assert!(write_jsonl(&path, &samples).is_err());
            assert_eq!(fs::read_to_string(&path).unwrap(), data);
            fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn explanatory_quoted_opening_think_tag_is_allowed() {
        let mut backend = teacher();
        backend.answer.text = "The literal '<think>' opens a reasoning block.".into();
        let mut prompt = prompt("quoted-opening");
        prompt.expected_answer = Some(backend.answer.text.clone());
        let samples = collect(&mut backend, &[prompt], &GenerationConfig::default()).unwrap();
        let directory = TestDirectory::new();
        let path = directory.path();
        write_jsonl(&path, &samples).unwrap();
        let samples = read_jsonl(&path).unwrap();
        assert_eq!(samples[0].response, backend.answer.text);
        let stats = train_samples(&mut model(8), &samples, 1, 0.001).unwrap();
        assert_eq!(stats.trained_bytes, backend.answer.text.len());
    }

    #[test]
    fn record_writer_enforces_serialization_budget_before_forwarding_bytes() {
        let mut output = Vec::new();
        let mut writer = RecordWriter {
            inner: &mut output,
            remaining: 8,
        };
        writer.write_all(b"1234").unwrap();
        assert!(writer.write_all(b"56789").is_err());
        assert_eq!(writer.remaining, 4);
        writer.write_all(b"5678").unwrap();
        assert!(writer.write_all(b"9").is_err());
        assert_eq!(output, b"12345678");

        let mut output = Vec::new();
        let error = serde_json::to_writer(
            &mut RecordWriter {
                inner: &mut output,
                remaining: 8,
            },
            &"\n".repeat(8),
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds"));
        assert!(output.len() <= 8);
    }

    #[test]
    fn reader_validates_records_before_reading_later_oversized_data() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let mut invalid = sample();
        invalid.response = "<think>unfinished".into();
        let valid = serde_json::to_string(&sample()).unwrap();
        for (prefix, expected_error) in [
            (
                serde_json::to_string(&invalid).unwrap() + "\n",
                "raw reasoning markers",
            ),
            (format!("{valid}\n{valid}\n"), "duplicate sample ID"),
        ] {
            fs::write(&path, prefix + &"x".repeat(MAX_LINE_BYTES + 1)).unwrap();
            let error = read_jsonl(&path).unwrap_err();
            assert!(error.contains(expected_error), "{error}");
        }
    }

    #[test]
    fn collection_preflights_all_prompts_and_generation_before_calls() {
        let mut backend = teacher();
        let mut invalid = prompt("bad");
        invalid.messages.clear();
        for prompts in [
            vec![],
            vec![prompt(" ")],
            vec![prompt("p"), prompt("p")],
            vec![prompt("p"), invalid],
        ] {
            assert!(collect(&mut backend, &prompts, &GenerationConfig::default()).is_err());
            assert_eq!(backend.calls, 0);
        }
        let options = GenerationConfig {
            temperature: f32::NAN,
            ..GenerationConfig::default()
        };
        assert!(collect(&mut backend, &[prompt("p")], &options).is_err());
        assert_eq!(backend.calls, 0);
        let value = serde_json::json!({"id":"p", "messages":[{"role":"user", "content":"x"}]});
        assert!(
            serde_json::from_value::<DistillPrompt>(value.clone())
                .unwrap()
                .expected_answer
                .is_none()
        );
        let mut unknown = value;
        unknown["typo"] = true.into();
        assert!(serde_json::from_value::<DistillPrompt>(unknown).is_err());
    }

    #[test]
    fn jsonl_roundtrip_never_overwrites_and_accepts_distinct_teachers() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let first = sample();
        let mut second = first.clone();
        second.teacher = "lfm".into();
        write_jsonl(&path, &[first.clone(), second]).unwrap();
        let original = fs::read(&path).unwrap();
        let loaded = read_jsonl(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(
            serde_json::to_value(&loaded[0]).unwrap(),
            serde_json::to_value(&first).unwrap()
        );
        assert!(write_jsonl(&path, &[first]).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
        // Concatenation and a missing final newline are both supported.
        fs::write(&path, &original[..original.len() - 1]).unwrap();
        assert_eq!(read_jsonl(&path).unwrap().len(), 2);
    }

    #[test]
    fn writer_errors_leave_no_partial_dataset_or_temporary_file() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let mut invalid = sample();
        invalid.format_version = 99;
        assert!(write_jsonl(&path, &[sample(), invalid]).is_err());
        assert!(!path.exists());
        let mut oversized = sample();
        oversized.response = "x".repeat(MAX_LINE_BYTES);
        assert!(write_jsonl(&path, &[sample(), oversized]).is_err());
        assert!(!path.exists());
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
    }

    #[test]
    fn reader_rejects_bad_records_duplicates_unknown_fields_and_oversized_lines() {
        let directory = TestDirectory::new();
        let path = directory.path();
        let valid = serde_json::to_string(&sample()).unwrap();
        for data in [
            String::new(),
            "\n".into(),
            "not json\n".into(),
            format!("{valid}\n\n"),
            format!("{valid}\n{valid}\n"),
        ] {
            fs::write(&path, data).unwrap();
            assert!(read_jsonl(&path).is_err());
        }
        for (field, value) in [
            ("format_version", serde_json::json!(2)),
            ("id", serde_json::json!(" ")),
            ("teacher", serde_json::json!("")),
            ("response", serde_json::json!("\n")),
            ("finish_reason", serde_json::json!("length")),
            ("reasoning_content", serde_json::json!("not a final answer")),
        ] {
            let mut record = serde_json::to_value(sample()).unwrap();
            record[field] = value;
            fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
            assert!(read_jsonl(&path).is_err(), "field {field}");
        }
        fs::write(&path, vec![b'x'; MAX_LINE_BYTES + 1]).unwrap();
        assert!(read_jsonl(&path).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn answer_inputs_are_causal_prompt_conditioned_and_target_only_response_bytes() {
        let model = model(128);
        let mut first = sample();
        first.response = "é!".into();
        let mut second = first.clone();
        second.teacher = "lfm".into();
        let mut third = first.clone();
        third.teacher = "third".into();
        let samples = [first, second, third];
        let mut targets = Vec::new();
        let mut streams = Vec::new();
        visit_targets(model.config(), &samples, |input, target| {
            let index = targets.len();
            let sample_index = index / 3;
            let offset = index % 3;
            let mut expected = prompt_bytes(&samples[sample_index].messages);
            expected.extend_from_slice(&samples[sample_index].response.as_bytes()[..offset]);
            assert_eq!(input.tokens[0].token, BOS);
            assert_eq!(
                input
                    .tokens
                    .iter()
                    .skip(1)
                    .map(|t| t.token as u8)
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(input.queries, vec![input.tokens.len() - 1]);
            for (row, mask) in input.visible.iter().enumerate() {
                for (col, &visible) in mask.iter().enumerate() {
                    assert_eq!(visible, col <= row);
                }
            }
            for token in &input.tokens {
                assert_eq!(token.kind, 0);
                assert_eq!(token.stream, sample_index % 2);
                assert!(token.stream < model.config().max_streams - 1);
            }
            targets.push(target as u8);
            streams.push(input.tokens[0].stream);
            Ok(())
        })
        .unwrap();
        assert_eq!(targets, "é!é!é!".as_bytes());
        assert_eq!(streams, [0, 0, 0, 1, 1, 1, 0, 0, 0]);
        let changed = answer_input(model.config(), b"a different prompt", &[], 0);
        let original = answer_input(model.config(), &prompt_bytes(&samples[0].messages), &[], 0);
        assert_ne!(
            model.predict(&changed).unwrap(),
            model.predict(&original).unwrap()
        );
    }

    #[test]
    fn sliding_context_keeps_every_long_response_target() {
        let model = model(4);
        let mut sample = sample();
        sample.response = "0123456789é".into();
        let mut targets = Vec::new();
        visit_targets(model.config(), &[sample.clone()], |input, target| {
            let offset = targets.len();
            let mut history = prompt_bytes(&sample.messages);
            history.extend_from_slice(&sample.response.as_bytes()[..offset]);
            assert_eq!(input.tokens.len(), 4);
            assert_eq!(
                input
                    .tokens
                    .iter()
                    .skip(1)
                    .map(|t| t.token as u8)
                    .collect::<Vec<_>>(),
                history[history.len() - 3..]
            );
            targets.push(target as u8);
            Ok(())
        })
        .unwrap();
        assert_eq!(targets, sample.response.as_bytes());
    }

    #[test]
    fn invalid_training_never_changes_predictions() {
        let mut model = model(8);
        let input = answer_input(model.config(), b"prompt", &[], 0);
        let before = model.predict(&input).unwrap();
        let valid = sample();
        let mut invalid = valid.clone();
        invalid.finish_reason = "length".into();
        assert!(train_samples(&mut model, &[valid.clone(), invalid], 1, 0.01).is_err());
        assert!(train_samples(&mut model, &[valid.clone(), valid.clone()], 1, 0.01).is_err());
        assert!(train_samples(&mut model, &[], 1, 0.01).is_err());
        assert!(train_samples(&mut model, std::slice::from_ref(&valid), 0, 0.01).is_err());
        assert!(train_samples(&mut model, std::slice::from_ref(&valid), usize::MAX, 0.01).is_err());
        for rate in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(train_samples(&mut model, std::slice::from_ref(&valid), 1, rate).is_err());
        }
        assert_eq!(model.predict(&input).unwrap(), before);
        let mut no_neutral = TinyTransformer::new(
            ModelConfig {
                max_streams: 1,
                ..model.config().clone()
            },
            42,
        )
        .unwrap();
        assert!(train_samples(&mut no_neutral, &[valid], 1, 0.01).is_err());
    }

    #[test]
    fn width_eight_training_reduces_loss_and_predicts_with_same_format() {
        let mut model = model(16);
        let mut sample = sample();
        sample.response = "a".into();
        let stats = train_samples(&mut model, &[sample.clone()], 80, 0.05).unwrap();
        assert_eq!(stats.trained_bytes, 80);
        assert!(stats.final_loss < stats.initial_loss * 0.25, "{stats:?}");
        assert_eq!(predict_answer(&model, &sample.messages, 1).unwrap(), "a");
        assert!(predict_answer(&model, &[], 1).is_err());
        assert!(predict_answer(&model, &sample.messages, 0).is_err());
    }
}
