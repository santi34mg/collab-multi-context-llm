//! Shared inference interface for native GGUF models and local debug servers.
//!
//! Teacher models retain their own tokenizers. Distillation transfers generated
//! text, never token IDs or logits between incompatible vocabularies.

use serde::{Deserialize, Serialize};

pub mod http;
#[cfg(feature = "native")]
pub mod native;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    pub stop: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.2,
            top_p: 0.95,
            seed: 42,
            stop: Vec::new(),
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_tokens == 0 || self.max_tokens > i32::MAX as usize {
            return Err("max_tokens must be in 1..=i32::MAX".into());
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be finite and nonnegative".into());
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.stop.iter().any(String::is_empty) {
            return Err("stop strings must be nonempty".into());
        }
        Ok(())
    }
}

pub fn validate_messages(messages: &[ChatMessage]) -> Result<(), String> {
    if messages.is_empty() {
        return Err("at least one chat message is required".into());
    }
    for message in messages {
        if !["system", "user", "assistant"].contains(&message.role.as_str()) {
            return Err(format!("unsupported chat role {:?}", message.role));
        }
        if message.content.trim().is_empty() {
            return Err("chat message content cannot be empty".into());
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Generation {
    pub text: String,
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

pub trait InferenceBackend {
    fn model_id(&self) -> &str;
    /// Raw, untemplated completion is available only on backends that support it.
    fn complete_raw(
        &mut self,
        _prompt: &str,
        _options: &GenerationConfig,
    ) -> Result<Generation, String> {
        Err("raw completion is unsupported by this backend".into())
    }
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        options: &GenerationConfig,
    ) -> Result<Generation, String>;
}
