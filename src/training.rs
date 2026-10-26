//! Supervised teacher forcing over executable, byte-framed operation traces.
//!
//! This is supervised trace infrastructure, not reinforcement learning: targets
//! come from demonstrations, never rewards or sampled model actions. Every
//! episode is replayed and context-checked in full before the first SGD update.

use crate::context::{self, ContextBudget};
use crate::memory::{MemoryDraft, MemoryStore};
use crate::model::{ModelConfig, ModelInput, TinyTransformer};
use crate::state::{Engine, EngineConfig, Operation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TrainingEpisode {
    pub scope: String,
    /// One list of initial working entries per stream, in stream-ID order.
    pub working: Vec<Vec<String>>,
    #[serde(default)]
    pub memory: Vec<MemoryDraft>,
    /// Each group starts together; shorter frames wait for their peers.
    pub rounds: Vec<Vec<TrainingAction>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TrainingAction {
    pub stream: usize,
    pub operation: Operation,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TrainingStats {
    /// Token-weighted mean cross entropy, measured before each SGD update.
    pub mean_loss: f32,
    pub predicted_tokens: usize,
    /// Byte-decoding rounds, not the number of logical action groups.
    pub rounds: u64,
}

/// Validate all initialization, byte transitions, operations, and contexts
/// without updating a model. Partial traces are allowed; finishing is optional.
pub fn validate_episode(
    episode: &TrainingEpisode,
    config: &ModelConfig,
    budget: &ContextBudget,
) -> Result<(), String> {
    replay(episode, config, budget, |_, _| Ok(())).map(|_| ())
}

/// Train on the next byte at each participating stream's query row, then advance
/// the engine with those same teacher bytes (not with model predictions).
///
/// Malformed demonstrations, including late operation or context failures, are
/// rejected before any updates. A numerical training failure can still occur
/// after earlier successful steps; this function is not an optimizer transaction.
pub fn train_episode(
    model: &mut TinyTransformer,
    episode: &TrainingEpisode,
    learning_rate: f32,
    budget: &ContextBudget,
) -> Result<TrainingStats, String> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive".into());
    }
    let config = model.config().clone();
    validate_episode(episode, &config, budget)?;

    let mut weighted_loss = 0.0_f64;
    let mut predicted_tokens = 0;
    let engine = replay(episode, &config, budget, |input, targets| {
        let loss = model.train_step(input, targets, learning_rate)?;
        weighted_loss += f64::from(loss) * targets.len() as f64;
        predicted_tokens += targets.len();
        Ok(())
    })?;
    Ok(TrainingStats {
        mean_loss: (weighted_loss / predicted_tokens as f64) as f32,
        predicted_tokens,
        rounds: engine.round(),
    })
}

/// Construct the task's starting state without applying demonstration rounds.
/// Also used for free-running inference on the same task format.
pub fn initial_engine(episode: &TrainingEpisode) -> Result<Engine, String> {
    initial_engine_with_memory(episode, MemoryStore::new(1024 * 1024))
}

/// Start a session with an existing backing store. Episode seed memories are
/// appended, not deduplicated; scope permissions remain fixed by the episode.
pub fn initial_engine_with_memory(
    episode: &TrainingEpisode,
    mut memory: MemoryStore,
) -> Result<Engine, String> {
    for (index, draft) in episode.memory.iter().enumerate() {
        if draft.scope != episode.scope {
            return Err(format!(
                "initial memory {index} scope differs from episode scope"
            ));
        }
        memory
            .commit(draft.clone())
            .map_err(|e| format!("initial memory {index}: {e}"))?;
    }
    let mut engine = Engine::new(
        EngineConfig {
            streams: episode.working.len(),
            scope: episode.scope.clone(),
            ..EngineConfig::default()
        },
        memory,
    )?;
    for (stream, entries) in episode.working.iter().enumerate() {
        for content in entries {
            engine
                .seed_entry(stream, content.clone())
                .map_err(|e| format!("initial working context for stream {stream}: {e}"))?;
        }
    }
    Ok(engine)
}

/// One replay path keeps preflight and training snapshot semantics identical.
fn replay(
    episode: &TrainingEpisode,
    config: &ModelConfig,
    budget: &ContextBudget,
    mut observe: impl FnMut(&ModelInput, &[usize]) -> Result<(), String>,
) -> Result<Engine, String> {
    let mut engine = initial_engine(episode)?;
    let mut has_targets = false;
    for (group, actions) in episode.rounds.iter().enumerate() {
        let mut frames = BTreeMap::new();
        for action in actions {
            let stream = engine
                .streams()
                .get(&action.stream)
                .ok_or_else(|| format!("group {group}: unknown stream {}", action.stream))?;
            if stream.is_finished() {
                return Err(format!(
                    "group {group}: stream {} has finished",
                    action.stream
                ));
            }
            if frames
                .insert(action.stream, action.operation.frame())
                .is_some()
            {
                return Err(format!(
                    "group {group}: multiple actions for stream {}",
                    action.stream
                ));
            }
        }
        let length = frames.values().map(Vec::len).max().unwrap_or(0);
        for offset in 0..length {
            let location = format!("group {group}, byte {offset}");
            let mut snapshot =
                context::encode(&engine, config, budget).map_err(|e| format!("{location}: {e}"))?;
            let mut queries = Vec::new();
            let mut targets = Vec::new();
            let mut tokens = Vec::new();
            // encode includes idle active streams. Filter only the query list;
            // token indices and the full visibility mask must remain intact.
            for (stream, query) in snapshot.streams.iter().zip(&snapshot.input.queries) {
                if let Some(&byte) = frames.get(stream).and_then(|frame| frame.get(offset)) {
                    queries.push(*query);
                    targets.push(usize::from(byte));
                    tokens.push((*stream, byte));
                }
            }
            snapshot.input.queries = queries;
            observe(&snapshot.input, &targets).map_err(|e| format!("{location}: {e}"))?;
            let report = engine
                .step_tokens(tokens)
                .map_err(|e| format!("{location}: {e}"))?;
            for outcome in report.outcomes {
                if !outcome.accepted {
                    return Err(format!(
                        "{location}: teacher operation for stream {} rejected: {}",
                        outcome.stream, outcome.detail
                    ));
                }
            }
            has_targets = true;
        }
    }
    if !has_targets {
        return Err("episode contains no teacher bytes".into());
    }
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryKind;
    use crate::state::TraceEvent;

    fn config() -> ModelConfig {
        ModelConfig {
            width: 8,
            max_positions: 512,
            max_streams: 4,
            context_kinds: 5,
        }
    }

    fn action(stream: usize, operation: Operation) -> TrainingAction {
        TrainingAction { stream, operation }
    }

    fn finish(stream: usize, result: &str) -> TrainingAction {
        action(
            stream,
            Operation::Finish {
                result: result.into(),
            },
        )
    }

    fn trivial() -> TrainingEpisode {
        TrainingEpisode {
            scope: "test".into(),
            working: vec![vec![]],
            memory: vec![],
            rounds: vec![vec![finish(0, "x")]],
        }
    }

    fn trace_loss(model: &TinyTransformer, episode: &TrainingEpisode) -> f64 {
        let mut loss = 0.0;
        let mut count = 0;
        replay(
            episode,
            model.config(),
            &ContextBudget::default(),
            |input, targets| {
                for (probabilities, &target) in model.predict(input)?.iter().zip(targets) {
                    loss -= f64::from(probabilities[target]).ln();
                    count += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        loss / count as f64
    }

    #[test]
    fn teacher_forcing_reduces_real_next_byte_loss() {
        let episode = trivial();
        let mut model = TinyTransformer::new(config(), 7).unwrap();
        let before = trace_loss(&model, &episode);
        for _ in 0..12 {
            let stats =
                train_episode(&mut model, &episode, 0.04, &ContextBudget::default()).unwrap();
            let bytes = episode.rounds[0][0].operation.frame().len();
            assert_eq!(stats.predicted_tokens, bytes);
            assert_eq!(stats.rounds, bytes as u64);
            assert!(stats.mean_loss.is_finite());
        }
        let after = trace_loss(&model, &episode);
        assert!(after < before * 0.8, "next-byte loss: {before} -> {after}");
    }

    #[test]
    fn malformed_demonstrations_never_update_predictions() {
        let mut model = TinyTransformer::new(config(), 11).unwrap();
        let probe = context::encode(
            &initial_engine(&trivial()).unwrap(),
            model.config(),
            &ContextBudget::default(),
        )
        .unwrap()
        .input;
        let before = model.predict(&probe).unwrap();
        let mut late_rejection = trivial();
        late_rejection.rounds = vec![
            vec![action(
                0,
                Operation::Publish {
                    target: None,
                    content: "valid first".into(),
                },
            )],
            vec![action(0, Operation::AcceptEdit { proposal_id: 999 })],
        ];
        let mut duplicate = trivial();
        duplicate.rounds = vec![vec![finish(0, "a"), finish(0, "b")]];
        let mut unknown = trivial();
        unknown.rounds.push(vec![finish(2, "unknown")]);
        let mut finished = trivial();
        finished.rounds.push(vec![finish(0, "again")]);
        let mut wrong_scope = trivial();
        wrong_scope.memory.push(MemoryDraft {
            content: "fact".into(),
            kind: MemoryKind::Fact,
            scope: "test ".into(),
            source_references: vec!["fixture".into()],
        });
        for episode in [late_rejection, duplicate, unknown, finished, wrong_scope] {
            assert!(train_episode(&mut model, &episode, 0.04, &ContextBudget::default()).is_err());
            assert_eq!(model.predict(&probe).unwrap(), before);
        }
        // Initial context fits, but a later byte snapshot exceeds the budget.
        let budget = ContextBudget {
            max_tokens: 4,
            ..ContextBudget::default()
        };
        assert!(train_episode(&mut model, &trivial(), 0.04, &budget).is_err());
        assert_eq!(model.predict(&probe).unwrap(), before);
    }

    #[test]
    fn peers_decode_simultaneously_and_short_frames_wait() {
        let short = action(
            1,
            Operation::Publish {
                target: None,
                content: "a".into(),
            },
        );
        let long = action(
            0,
            Operation::Publish {
                target: None,
                content: "a longer publication".into(),
            },
        );
        let short_frame = short.operation.frame();
        let long_frame = long.operation.frame();
        let episode = TrainingEpisode {
            scope: "test".into(),
            working: vec![vec![], vec![], vec![]],
            memory: vec![],
            // Reverse stream order checks that targets follow encoded query order.
            rounds: vec![vec![short, long]],
        };
        let mut round = 0;
        let engine = replay(
            &episode,
            &config(),
            &ContextBudget::default(),
            |input, targets| {
                let expected = if round < short_frame.len() {
                    vec![long_frame[round] as usize, short_frame[round] as usize]
                } else {
                    vec![long_frame[round] as usize]
                };
                assert_eq!(targets, expected);
                assert_eq!(input.queries.len(), expected.len());
                assert_eq!(input.tokens[input.queries[0]].stream, 0);
                for &query in &input.queries {
                    let frame = if input.tokens[query].stream == 0 {
                        &long_frame
                    } else {
                        &short_frame
                    };
                    let previous = if round == 0 {
                        crate::model::BOS
                    } else {
                        usize::from(frame[round - 1])
                    };
                    assert_eq!(input.tokens[query].token, previous);
                }
                // Both waiting and never-acting streams retain their context tokens.
                assert!(input.tokens.iter().any(|token| token.stream == 1));
                assert!(input.tokens.iter().any(|token| token.stream == 2));
                assert_eq!(input.visible.len(), input.tokens.len());
                assert!(
                    input
                        .visible
                        .iter()
                        .all(|row| row.len() == input.tokens.len())
                );
                round += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(engine.round(), long_frame.len() as u64);
        assert_eq!(engine.streams()[&0].history(), long_frame);
        assert_eq!(engine.streams()[&1].history(), short_frame);
        assert!(engine.streams()[&2].history().is_empty());
        for event in engine.trace() {
            if let TraceEvent::Token {
                round,
                stream,
                byte,
            } = event
            {
                let frame = if *stream == 0 {
                    &long_frame
                } else {
                    &short_frame
                };
                assert_eq!(*byte, frame[*round as usize]);
            }
        }
    }

    #[test]
    fn mean_loss_is_weighted_by_supervised_bytes_not_rounds() {
        let mut episode = trivial();
        episode.working.push(vec![]);
        episode.rounds[0].push(finish(1, "a longer result"));
        let mut actual = TinyTransformer::new(config(), 3).unwrap();
        let mut reference = actual.clone();
        let mut sum = 0.0_f64;
        let mut count = 0;
        let engine = replay(
            &episode,
            &config(),
            &ContextBudget::default(),
            |input, targets| {
                sum +=
                    f64::from(reference.train_step(input, targets, 0.02)?) * targets.len() as f64;
                count += targets.len();
                Ok(())
            },
        )
        .unwrap();
        let stats = train_episode(&mut actual, &episode, 0.02, &ContextBudget::default()).unwrap();
        assert_eq!(stats.mean_loss, (sum / count as f64) as f32);
        assert_eq!(stats.predicted_tokens, count);
        assert_eq!(stats.rounds, engine.round());
    }

    #[test]
    fn memory_is_committed_scoped_and_retrievable() {
        let mut episode = trivial();
        episode.memory.push(MemoryDraft {
            content: "known fact".into(),
            kind: MemoryKind::Fact,
            scope: episode.scope.clone(),
            source_references: vec!["fixture".into()],
        });
        episode.rounds.insert(
            0,
            vec![action(
                0,
                Operation::MemoryQuery {
                    query: "known".into(),
                },
            )],
        );
        let engine = replay(
            &episode,
            &config(),
            &ContextBudget::default(),
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(engine.streams()[&0].retrieved().len(), 1);
        assert_eq!(engine.streams()[&0].retrieved()[0].content, "known fact");
    }

    #[test]
    fn dataset_serde_defaults_memory_and_rejects_unknown_fields() {
        let json = r#"[{"scope":"test","working":[[]],"rounds":[[{"stream":0,"operation":{"op":"finish","result":"x"}}]]}]"#;
        let episodes: Vec<TrainingEpisode> = serde_json::from_str(json).unwrap();
        assert!(episodes[0].memory.is_empty());
        let serialized = serde_json::to_string(&episodes).unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<TrainingEpisode>>(&serialized).unwrap()[0].scope,
            "test"
        );
        assert!(
            serde_json::from_str::<Vec<TrainingEpisode>>(
                &json.replace("\"scope\":", "\"unknown\":0,\"scope\":")
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<Vec<TrainingEpisode>>(
                &json.replace("\"stream\":", "\"unknown\":0,\"stream\":")
            )
            .is_err()
        );
    }
}
