//! Turn an engine snapshot into structured token blocks and a visibility mask.
//! The reference implementation rebuilds every round: no stale KV caches.

use crate::model::{BOS, InputToken, ModelConfig, ModelInput, TinyTransformer};
use crate::state::{Engine, ProposalStatus, StreamId};
use std::collections::BTreeMap;

pub const HISTORY: usize = 0;
pub const WORKING: usize = 1;
pub const SHARED: usize = 2;
pub const MEMORY: usize = 3;
pub const INBOX: usize = 4;
pub const CONTEXT_KINDS: usize = 5;

#[derive(Clone, Debug)]
pub struct ContextBudget {
    pub max_tokens: usize,
    /// Only model visibility is truncated; the engine's audit history is intact.
    pub history_bytes: usize,
    /// Ablation: disable peer publications without changing the engine.
    pub read_publications: bool,
    /// Ablation: disable retrieved memory without deleting it.
    pub read_memory: bool,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            history_bytes: 256,
            read_publications: true,
            read_memory: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodedSnapshot {
    pub input: ModelInput,
    /// Corresponds exactly to the model's output/query order.
    pub streams: Vec<StreamId>,
}

struct Builder<'a> {
    config: &'a ModelConfig,
    budget: &'a ContextBudget,
    tokens: Vec<InputToken>,
    readers: Vec<Vec<StreamId>>,
    positions: BTreeMap<(StreamId, StreamId, usize), usize>,
}

impl Builder<'_> {
    fn block(
        &mut self,
        bytes: &[u8],
        owner: usize,
        kind: usize,
        readers: Vec<usize>,
        bos: bool,
    ) -> Result<usize, String> {
        let size = bytes
            .len()
            .checked_add(usize::from(bos))
            .ok_or("context size overflow")?;
        if readers.len() != 1 {
            return Err("reference blocks require exactly one reader".into());
        }
        let key = (readers[0], owner, kind);
        let offset = self.positions.get(&key).copied().unwrap_or(0);
        if size > self.config.max_positions.saturating_sub(offset) {
            return Err(format!(
                "context window needs {} positions; model has {}",
                offset.saturating_add(size),
                self.config.max_positions
            ));
        }
        if size > self.budget.max_tokens.saturating_sub(self.tokens.len()) {
            return Err(
                "encoded snapshot exceeds total token budget; reduce contexts or retrieval".into(),
            );
        }
        let start = self.tokens.len();
        for (position, token) in bos
            .then_some(BOS)
            .into_iter()
            .chain(bytes.iter().map(|b| *b as usize))
            .enumerate()
        {
            self.tokens.push(InputToken {
                token,
                position: offset + position,
                stream: owner,
                kind,
            });
            self.readers.push(readers.clone());
        }
        self.positions.insert(key, offset + size);
        Ok(start)
    }
}

pub fn encode(
    engine: &Engine,
    config: &ModelConfig,
    budget: &ContextBudget,
) -> Result<EncodedSnapshot, String> {
    let streams = engine.active_streams();
    if streams.is_empty() {
        return Err("no active streams to encode".into());
    }
    // Memory has a neutral identity; publications retain their author's identity.
    let neutral = engine.config().streams;
    if config.max_streams <= neutral
        || config.context_kinds < CONTEXT_KINDS
        || config.max_positions == 0
    {
        return Err("model needs one identity per stream plus a neutral identity, five context kinds, and positions".into());
    }
    if budget.max_tokens == 0 {
        return Err("context token budget must be positive".into());
    }
    let mut builder = Builder {
        config,
        budget,
        tokens: Vec::new(),
        readers: Vec::new(),
        positions: BTreeMap::new(),
    };
    let mut queries = Vec::new();
    for stream in engine.streams().values() {
        if stream.is_finished() {
            continue;
        }
        for entry in stream.working().values() {
            // Identity and version are visible so generated edits can address
            // stable entries; only current content participates in attention.
            let version = entry.current();
            let text = format!(
                "entry {} v{}: {}",
                entry.id, version.version, version.content
            );
            builder.block(
                text.as_bytes(),
                stream.id(),
                WORKING,
                vec![stream.id()],
                false,
            )?;
        }
        for proposal in engine
            .proposals()
            .values()
            .filter(|p| p.target == stream.id() && p.status == ProposalStatus::Pending)
        {
            let text = serde_json::to_vec(proposal).map_err(|e| e.to_string())?;
            builder.block(&text, stream.id(), INBOX, vec![stream.id()], false)?;
        }
        for notice in stream.notices() {
            builder.block(
                notice.as_bytes(),
                stream.id(),
                INBOX,
                vec![stream.id()],
                false,
            )?;
        }
        if budget.read_memory {
            for memory in stream.retrieved() {
                let text = serde_json::to_vec(memory).map_err(|e| e.to_string())?;
                builder.block(&text, neutral, MEMORY, vec![stream.id()], false)?;
            }
        }
        let history = stream.history();
        let retained = budget
            .history_bytes
            .min(config.max_positions - 1)
            .min(history.len());
        let start = builder.block(
            &history[history.len() - retained..],
            stream.id(),
            HISTORY,
            vec![stream.id()],
            true,
        )?;
        queries.push(start + retained);
    }
    if budget.read_publications {
        for publication in engine.workspace() {
            let readers = match publication.target {
                Some(target) => {
                    let mut readers = vec![publication.author, target];
                    readers.sort_unstable();
                    readers.dedup();
                    readers
                }
                None => streams.clone(),
            };
            // Source identity and visibility are structural, not prose the
            // model must infer from a serialized envelope. Publication IDs are
            // not operation targets, so only the payload needs tokenization.
            // Per-reader blocks keep offsets independent of publications that
            // reader cannot see. Invisible messages consume no context budget.
            for reader in readers
                .into_iter()
                .filter(|reader| streams.contains(reader))
            {
                builder.block(
                    publication.content.as_bytes(),
                    publication.author,
                    SHARED,
                    vec![reader],
                    false,
                )?;
            }
        }
    }
    let size = builder.tokens.len();
    // The tiny single-layer model only evaluates query rows. Other rows have
    // identity masks; this must be revisited before stacking decoder layers.
    let mut visible = vec![vec![false; size]; size];
    for (row, values) in visible.iter_mut().enumerate() {
        values[row] = true;
    }
    for (&query, &stream) in queries.iter().zip(&streams) {
        for (key, readers) in builder.readers.iter().enumerate() {
            visible[query][key] = readers.contains(&stream);
        }
    }
    Ok(EncodedSnapshot {
        input: ModelInput {
            tokens: builder.tokens,
            visible,
            queries,
        },
        streams,
    })
}

/// Greedy decoding emits at most one byte per active stream from one snapshot.
/// BOS is input-only and excluded from the generated-byte argmax.
pub fn predict_round(
    engine: &Engine,
    model: &TinyTransformer,
    budget: &ContextBudget,
) -> Result<Vec<(StreamId, u8)>, String> {
    let snapshot = encode(engine, model.config(), budget)?;
    let probabilities = model.predict(&snapshot.input)?;
    Ok(snapshot
        .streams
        .into_iter()
        .zip(probabilities)
        .map(|(stream, distribution)| {
            let token = (0..256)
                .max_by(|&a, &b| {
                    distribution[a]
                        .total_cmp(&distribution[b])
                        .then_with(|| b.cmp(&a))
                })
                .expect("byte vocabulary");
            (stream, token as u8)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;
    use crate::state::{EngineConfig, Operation};

    fn config() -> ModelConfig {
        ModelConfig {
            width: 8,
            max_positions: 512,
            max_streams: 4,
            context_kinds: 5,
        }
    }

    #[test]
    fn private_streams_do_not_leak_into_each_others_queries() {
        let mut engine = Engine::new(EngineConfig::default(), MemoryStore::new(100)).unwrap();
        engine.seed_entry(0, "secret A").unwrap();
        engine.seed_entry(1, "secret B").unwrap();
        engine.step_tokens(vec![(0, b'x'), (1, b'y')]).unwrap();
        let input = encode(&engine, &config(), &ContextBudget::default())
            .unwrap()
            .input;
        for (stream, &query) in input.queries.iter().enumerate() {
            for (key, token) in input.tokens.iter().enumerate() {
                assert_eq!(input.visible[query][key], token.stream == stream);
            }
        }
    }

    #[test]
    fn targeted_publications_are_visible_only_to_sender_and_recipient() {
        let mut engine = Engine::new(
            EngineConfig {
                streams: 3,
                ..EngineConfig::default()
            },
            MemoryStore::new(100),
        )
        .unwrap();
        engine
            .step_actions(vec![(
                0,
                Operation::Publish {
                    target: Some(1),
                    content: "answer".into(),
                },
            )])
            .unwrap();
        let input = encode(&engine, &config(), &ContextBudget::default())
            .unwrap()
            .input;
        for &query in &input.queries {
            let stream = input.tokens[query].stream;
            for (key, token) in input
                .tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| t.kind == SHARED)
            {
                assert_eq!(token.stream, 0);
                if stream == 2 {
                    assert!(!input.visible[query][key]);
                }
            }
            let visible_payload = input
                .tokens
                .iter()
                .enumerate()
                .filter(|(key, token)| token.kind == SHARED && input.visible[query][*key])
                .count();
            assert_eq!(visible_payload, if stream < 2 { 6 } else { 0 });
        }
    }

    #[test]
    fn history_truncation_is_explicit_and_does_not_erase_audit() {
        let mut engine = Engine::new(EngineConfig::default(), MemoryStore::new(100)).unwrap();
        for byte in b"abcdef" {
            engine.step_tokens(vec![(0, *byte)]).unwrap();
        }
        let snapshot = encode(
            &engine,
            &config(),
            &ContextBudget {
                history_bytes: 2,
                ..ContextBudget::default()
            },
        )
        .unwrap();
        let local: Vec<usize> = snapshot
            .input
            .tokens
            .iter()
            .filter(|t| t.stream == 0)
            .map(|t| t.token)
            .collect();
        assert_eq!(local, vec![BOS, b'e' as usize, b'f' as usize]);
        assert_eq!(engine.streams()[&0].history(), b"abcdef");
    }

    #[test]
    fn entry_positions_preserve_identity_content_association() {
        let mut engine = Engine::new(EngineConfig::default(), MemoryStore::new(100)).unwrap();
        engine.seed_entry(0, "A").unwrap();
        engine.seed_entry(0, "B").unwrap();
        let snapshot = encode(&engine, &config(), &ContextBudget::default()).unwrap();
        let working: Vec<_> = snapshot
            .input
            .tokens
            .iter()
            .filter(|t| t.kind == WORKING)
            .collect();
        assert!(
            working
                .windows(2)
                .all(|pair| pair[1].position == pair[0].position + 1)
        );
        let positions: Vec<_> = working
            .iter()
            .filter(|t| t.token == b'A' as usize || t.token == b'B' as usize)
            .map(|t| t.position)
            .collect();
        assert_eq!(positions.len(), 2);
        assert_ne!(positions[0], positions[1]);
    }

    #[test]
    fn invisible_publications_do_not_change_visible_positions_or_capacity() {
        let mut engine = Engine::new(
            EngineConfig {
                streams: 3,
                ..EngineConfig::default()
            },
            MemoryStore::new(100),
        )
        .unwrap();
        engine
            .step_actions(vec![(
                0,
                Operation::Publish {
                    target: Some(0),
                    content: "hidden".into(),
                },
            )])
            .unwrap();
        engine
            .step_actions(vec![(
                0,
                Operation::Publish {
                    target: Some(2),
                    content: "visible".into(),
                },
            )])
            .unwrap();
        let snapshot = encode(&engine, &config(), &ContextBudget::default()).unwrap();
        let query = snapshot.input.queries[2];
        let positions: Vec<_> = snapshot
            .input
            .tokens
            .iter()
            .enumerate()
            .filter(|(key, token)| token.kind == SHARED && snapshot.input.visible[query][*key])
            .map(|(_, token)| token.position)
            .collect();
        assert_eq!(positions, (0..7).collect::<Vec<_>>());
        engine
            .step_actions(vec![
                (
                    0,
                    Operation::Finish {
                        result: "done".into(),
                    },
                ),
                (
                    2,
                    Operation::Finish {
                        result: "done".into(),
                    },
                ),
            ])
            .unwrap();
        let snapshot = encode(
            &engine,
            &config(),
            &ContextBudget {
                max_tokens: 1,
                ..ContextBudget::default()
            },
        )
        .unwrap();
        assert_eq!(snapshot.input.tokens.len(), 1); // only stream 1's BOS
    }

    #[test]
    fn bounds_fail_instead_of_silently_dropping_working_context() {
        let mut engine = Engine::new(EngineConfig::default(), MemoryStore::new(100)).unwrap();
        engine.seed_entry(0, "context").unwrap();
        assert!(
            encode(
                &engine,
                &config(),
                &ContextBudget {
                    max_tokens: 1,
                    ..ContextBudget::default()
                }
            )
            .is_err()
        );
        let mut short = config();
        short.max_positions = 2;
        assert!(encode(&engine, &short, &ContextBudget::default()).is_err());
    }
}
