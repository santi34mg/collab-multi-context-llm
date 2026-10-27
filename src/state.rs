//! Deterministic snapshot rounds and the executable collaboration protocol.
//!
//! The host owns permissions and scope. Generated JSON can request operations,
//! but cannot change policy. State is changed only through this engine; callers
//! receive read-only views. Traces contain private data and are host diagnostics.

use crate::memory::{MemoryDraft, MemoryEntry, MemoryKind, MemoryStore};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type StreamId = usize;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub streams: usize,
    pub scope: String,
    pub max_content_bytes: usize,
    pub max_frame_bytes: usize,
    pub max_history_bytes: usize,
    pub max_workspace_bytes: usize,
    pub max_retrieved_entries: usize,
    pub max_retrieved_bytes: usize,
    pub max_pending_memory: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            streams: 2,
            scope: "default".into(),
            max_content_bytes: 1024,
            max_frame_bytes: 4096,
            max_history_bytes: 16384,
            max_workspace_bytes: 8192,
            max_retrieved_entries: 4,
            max_retrieved_bytes: 2048,
            max_pending_memory: 16,
        }
    }
}

/// Newline-delimited JSON is both the byte decoder's operation format and the
/// supervised trace format. Scope is deliberately not a model-selectable field.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Publish {
        target: Option<StreamId>,
        content: String,
    },
    ProposeEdit {
        target: StreamId,
        entry_id: u64,
        expected_version: u64,
        replacement: String,
    },
    AcceptEdit {
        proposal_id: u64,
    },
    RejectEdit {
        proposal_id: u64,
    },
    MemoryQuery {
        query: String,
    },
    MemoryPropose {
        content: String,
        kind: MemoryKind,
        source_references: Vec<String>,
    },
    MemoryRevise {
        entry_id: u64,
        expected_version: u64,
        content: String,
        kind: MemoryKind,
        source_references: Vec<String>,
    },
    Finish {
        result: String,
    },
}

impl Operation {
    pub fn frame(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(self).expect("operation serializes");
        bytes.push(b'\n');
        bytes
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EntryVersion {
    pub version: u64,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkingEntry {
    pub id: u64,
    pub versions: Vec<EntryVersion>,
}

impl WorkingEntry {
    pub fn current(&self) -> &EntryVersion {
        self.versions.last().expect("entries always have a version")
    }
}

#[derive(Clone, Debug)]
pub struct Stream {
    pub(crate) id: StreamId,
    pub(crate) working: BTreeMap<u64, WorkingEntry>,
    pub(crate) history: Vec<u8>,
    frame: Vec<u8>,
    discarding_frame: bool,
    pub(crate) retrieved: Vec<MemoryEntry>,
    pub(crate) notices: Vec<String>,
    pub(crate) result: Option<String>,
}

impl Stream {
    pub fn id(&self) -> StreamId {
        self.id
    }
    pub fn working(&self) -> &BTreeMap<u64, WorkingEntry> {
        &self.working
    }
    pub fn history(&self) -> &[u8] {
        &self.history
    }
    pub fn retrieved(&self) -> &[MemoryEntry] {
        &self.retrieved
    }
    pub fn notices(&self) -> &[String] {
        &self.notices
    }
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }
    pub fn is_finished(&self) -> bool {
        self.result.is_some()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Publication {
    pub id: u64,
    pub author: StreamId,
    pub target: Option<StreamId>,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Accepted,
    Rejected,
}

#[derive(Clone, Debug, Serialize)]
pub struct EditProposal {
    pub id: u64,
    pub author: StreamId,
    pub target: StreamId,
    pub entry_id: u64,
    pub expected_version: u64,
    pub replacement: String,
    pub status: ProposalStatus,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActionOutcome {
    pub stream: StreamId,
    pub operation: Option<Operation>,
    pub accepted: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoundReport {
    /// The snapshot round read by these outputs, starting at zero.
    pub round: u64,
    pub outcomes: Vec<ActionOutcome>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TraceEvent {
    Token {
        round: u64,
        stream: StreamId,
        byte: u8,
    },
    Action {
        round: u64,
        outcome: ActionOutcome,
    },
    MemoryCommit {
        round: u64,
        ids: Vec<u64>,
    },
}

#[derive(Clone, Debug)]
enum PendingMemory {
    New(MemoryDraft),
    Revise {
        id: u64,
        version: u64,
        draft: MemoryDraft,
    },
}

#[derive(Clone, Debug)]
pub struct Engine {
    config: EngineConfig,
    streams: BTreeMap<StreamId, Stream>,
    workspace: Vec<Publication>,
    proposals: BTreeMap<u64, EditProposal>,
    memory: MemoryStore,
    pending_memory: Vec<PendingMemory>,
    trace: Vec<TraceEvent>,
    round: u64,
    next_entry: u64,
    next_publication: u64,
    next_proposal: u64,
}

impl Engine {
    pub fn new(config: EngineConfig, memory: MemoryStore) -> Result<Self, String> {
        if config.streams == 0 || config.scope.trim().is_empty() {
            return Err("at least one stream and a nonempty scope are required".into());
        }
        if config.max_content_bytes == 0
            || config.max_frame_bytes == 0
            || config.max_history_bytes == 0
            || config.max_workspace_bytes == 0
        {
            return Err("content, frame, history, and workspace budgets must be positive".into());
        }
        let streams = (0..config.streams)
            .map(|id| {
                (
                    id,
                    Stream {
                        id,
                        working: BTreeMap::new(),
                        history: Vec::new(),
                        frame: Vec::new(),
                        discarding_frame: false,
                        retrieved: Vec::new(),
                        notices: Vec::new(),
                        result: None,
                    },
                )
            })
            .collect();
        Ok(Self {
            config,
            streams,
            workspace: Vec::new(),
            proposals: BTreeMap::new(),
            memory,
            pending_memory: Vec::new(),
            trace: Vec::new(),
            round: 0,
            next_entry: 1,
            next_publication: 1,
            next_proposal: 1,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
    pub fn round(&self) -> u64 {
        self.round
    }
    pub fn streams(&self) -> &BTreeMap<StreamId, Stream> {
        &self.streams
    }
    pub fn workspace(&self) -> &[Publication] {
        &self.workspace
    }
    pub fn proposals(&self) -> &BTreeMap<u64, EditProposal> {
        &self.proposals
    }
    pub fn memory(&self) -> &MemoryStore {
        &self.memory
    }
    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }
    pub fn is_finished(&self) -> bool {
        self.streams.values().all(Stream::is_finished)
    }
    pub fn active_streams(&self) -> Vec<StreamId> {
        self.streams
            .values()
            .filter(|s| !s.is_finished())
            .map(|s| s.id)
            .collect()
    }

    /// Host-only initialization, forbidden after the first decoding round.
    pub fn seed_entry(
        &mut self,
        stream: StreamId,
        content: impl Into<String>,
    ) -> Result<u64, String> {
        if self.round != 0 {
            return Err("working context can only be seeded before decoding".into());
        }
        let content = content.into();
        self.check_content(&content)?;
        let target = self.streams.get_mut(&stream).ok_or("unknown stream")?;
        let id = self.next_entry;
        self.next_entry = id.checked_add(1).ok_or("entry ID overflow")?;
        target.working.insert(
            id,
            WorkingEntry {
                id,
                versions: vec![EntryVersion {
                    version: 1,
                    content,
                }],
            },
        );
        Ok(id)
    }

    /// Apply scripted structured actions. Each actor may submit at most one.
    /// All eligibility checks use the old snapshot, never earlier actors' writes.
    /// Valid actions commit in ascending stream order, making conflicts explicit.
    pub fn step_actions(
        &mut self,
        mut actions: Vec<(StreamId, Operation)>,
    ) -> Result<RoundReport, String> {
        self.validate_actors(actions.iter().map(|(id, _)| *id))?;
        actions.sort_by_key(|(id, _)| *id);
        let round = self.round;
        let next_round = round.checked_add(1).ok_or("round overflow")?;
        // Retain validation results from the old snapshot before any writes,
        // rather than cloning growing histories, trace, and memory each round.
        let validated: Vec<_> = actions
            .into_iter()
            .map(|(stream, operation)| {
                let validation = self.validate_operation(stream, &operation);
                (stream, operation, validation)
            })
            .collect();
        let mut outcomes = Vec::new();
        for (stream, operation, validation) in validated {
            let result = validation.and_then(|()| self.apply_operation(stream, &operation));
            let (accepted, detail) = match result {
                Ok(detail) => (true, detail),
                Err(e) => (false, e),
            };
            let outcome = ActionOutcome {
                stream,
                operation: Some(operation),
                accepted,
                detail,
            };
            self.record_outcome(round, &outcome);
            outcomes.push(outcome);
        }
        self.round = next_round;
        Ok(RoundReport { round, outcomes })
    }

    /// A byte tokenizer makes the model vocabulary independent of dependencies.
    /// Only complete UTF-8 JSON lines execute. Oversized frames are discarded
    /// through the next newline so a suffix cannot become a second operation.
    pub fn step_tokens(&mut self, mut tokens: Vec<(StreamId, u8)>) -> Result<RoundReport, String> {
        self.validate_actors(tokens.iter().map(|(id, _)| *id))?;
        tokens.sort_by_key(|(id, _)| *id);
        // All round-level failures must be checked before appending any bytes.
        // Operation rejections below are outcomes, not transactional failures.
        for &(id, _) in &tokens {
            if self.streams[&id].history.len() >= self.config.max_history_bytes {
                return Err(format!(
                    "stream {id} exhausted its generation-history budget"
                ));
            }
        }
        self.round.checked_add(1).ok_or("round overflow")?;
        let round = self.round;
        let mut actions = Vec::new();
        let mut malformed = Vec::new();
        for (id, byte) in tokens {
            let stream = self.streams.get_mut(&id).expect("validated actor");
            stream.history.push(byte);
            self.trace.push(TraceEvent::Token {
                round,
                stream: id,
                byte,
            });
            if stream.discarding_frame {
                if byte == b'\n' {
                    stream.discarding_frame = false;
                }
                continue;
            }
            if byte != b'\n' {
                if stream.frame.len() >= self.config.max_frame_bytes {
                    stream.frame.clear();
                    stream.discarding_frame = true;
                    malformed.push((id, "operation frame exceeded its byte budget".to_string()));
                } else {
                    stream.frame.push(byte);
                }
                continue;
            }
            let frame = std::mem::take(&mut stream.frame);
            if frame.is_empty() {
                continue;
            }
            match serde_json::from_slice::<Operation>(&frame) {
                Ok(operation) => actions.push((id, operation)),
                Err(e) => malformed.push((id, format!("invalid operation frame: {e}"))),
            }
        }
        // Appended bytes are not editable state and do not affect operation
        // authorization. step_actions still validates all operations together.
        let mut report = self.step_actions(actions)?;
        for (stream, detail) in malformed {
            let outcome = ActionOutcome {
                stream,
                operation: None,
                accepted: false,
                detail,
            };
            self.record_outcome(round, &outcome);
            report.outcomes.push(outcome);
        }
        report.outcomes.sort_by_key(|outcome| outcome.stream);
        Ok(report)
    }

    /// All-or-nothing task-boundary memory commit. This validates structure and
    /// budget, not factual truth. The host must call MemoryStore::save separately.
    pub fn commit_memory(&mut self) -> Result<Vec<MemoryEntry>, String> {
        if !self.is_finished() {
            return Err("memory commits require all streams to finish".into());
        }
        let mut memory = self.memory.clone();
        let mut entries = Vec::new();
        for pending in &self.pending_memory {
            entries.push(match pending {
                PendingMemory::New(draft) => memory.commit(draft.clone())?,
                PendingMemory::Revise { id, version, draft } => {
                    memory.revise(*id, *version, draft.clone())?
                }
            });
        }
        self.memory = memory;
        self.pending_memory.clear();
        self.trace.push(TraceEvent::MemoryCommit {
            round: self.round,
            ids: entries.iter().map(|e| e.id).collect(),
        });
        Ok(entries)
    }

    fn validate_actors(&self, actors: impl Iterator<Item = StreamId>) -> Result<(), String> {
        if self.is_finished() {
            return Err("task is already finished".into());
        }
        let mut seen = BTreeSet::new();
        for id in actors {
            let stream = self
                .streams
                .get(&id)
                .ok_or_else(|| format!("unknown stream {id}"))?;
            if stream.is_finished() {
                return Err(format!("stream {id} has finished"));
            }
            if !seen.insert(id) {
                return Err(format!("multiple outputs for stream {id} in one round"));
            }
        }
        Ok(())
    }

    fn check_content(&self, text: &str) -> Result<(), String> {
        if text.trim().is_empty() {
            return Err("content cannot be empty".into());
        }
        if text.len() > self.config.max_content_bytes {
            return Err("content exceeds byte budget".into());
        }
        Ok(())
    }

    fn validate_operation(&self, actor: StreamId, operation: &Operation) -> Result<(), String> {
        match operation {
            Operation::Publish { target, content } => {
                self.check_content(content)?;
                if let Some(target) = target {
                    self.streams
                        .get(target)
                        .ok_or("unknown publication target")?;
                }
            }
            Operation::ProposeEdit {
                target,
                entry_id,
                expected_version,
                replacement,
            } => {
                self.check_content(replacement)?;
                if *target == actor {
                    return Err("peer edits must target another stream".into());
                }
                let target = self.streams.get(target).ok_or("unknown edit target")?;
                if target.is_finished() {
                    return Err("edit target has finished".into());
                }
                let entry = target
                    .working
                    .get(entry_id)
                    .ok_or("unknown working entry")?;
                if entry.current().version != *expected_version {
                    return Err("stale working-entry version".into());
                }
            }
            Operation::AcceptEdit { proposal_id } | Operation::RejectEdit { proposal_id } => {
                let proposal = self
                    .proposals
                    .get(proposal_id)
                    .ok_or("proposal is not in the committed snapshot")?;
                if proposal.target != actor {
                    return Err("only the target stream may decide an edit".into());
                }
                if proposal.status != ProposalStatus::Pending {
                    return Err("proposal is no longer pending".into());
                }
            }
            Operation::MemoryQuery { query } => {
                self.check_content(query)?;
            }
            Operation::MemoryPropose {
                content,
                kind,
                source_references,
            } => {
                let draft = self.memory_draft(actor, content, *kind, source_references)?;
                let mut validation = self.memory.clone();
                validation.commit(draft)?;
            }
            Operation::MemoryRevise {
                entry_id,
                expected_version,
                content,
                kind,
                source_references,
            } => {
                if !self
                    .memory
                    .active(&self.config.scope)
                    .iter()
                    .any(|entry| entry.id == *entry_id)
                {
                    return Err("memory entry is not active in this scope".into());
                }
                let draft = self.memory_draft(actor, content, *kind, source_references)?;
                let mut validation = self.memory.clone();
                validation.revise(*entry_id, *expected_version, draft)?;
            }
            Operation::Finish { result } => {
                self.check_content(result)?;
            }
        }
        Ok(())
    }

    fn memory_draft(
        &self,
        actor: StreamId,
        content: &str,
        kind: MemoryKind,
        sources: &[String],
    ) -> Result<MemoryDraft, String> {
        self.check_content(content)?;
        if sources.is_empty()
            || sources
                .iter()
                .any(|s| s.trim().is_empty() || s.len() > self.config.max_content_bytes)
            || sources.len() > 16
        {
            return Err("memory requires 1..=16 bounded, nonempty source references".into());
        }
        let mut source_references = sources.to_vec();
        source_references.push(format!("stream:{actor}/round:{}", self.round));
        Ok(MemoryDraft {
            content: content.into(),
            kind,
            scope: self.config.scope.clone(),
            source_references,
        })
    }

    fn apply_operation(
        &mut self,
        actor: StreamId,
        operation: &Operation,
    ) -> Result<String, String> {
        match operation {
            Operation::Publish { target, content } => {
                let used: usize = self.workspace.iter().map(|p| p.content.len()).sum();
                if content.len() > self.config.max_workspace_bytes.saturating_sub(used) {
                    return Err("shared workspace byte budget exhausted".into());
                }
                let id = self.next_publication;
                let next = id.checked_add(1).ok_or("publication ID overflow")?;
                self.workspace.push(Publication {
                    id,
                    author: actor,
                    target: *target,
                    content: content.clone(),
                });
                self.next_publication = next;
                Ok(format!("published {id}"))
            }
            Operation::ProposeEdit {
                target,
                entry_id,
                expected_version,
                replacement,
            } => {
                let id = self.next_proposal;
                let next = id.checked_add(1).ok_or("proposal ID overflow")?;
                self.proposals.insert(
                    id,
                    EditProposal {
                        id,
                        author: actor,
                        target: *target,
                        entry_id: *entry_id,
                        expected_version: *expected_version,
                        replacement: replacement.clone(),
                        status: ProposalStatus::Pending,
                    },
                );
                self.next_proposal = next;
                Ok(format!("proposed edit {id}"))
            }
            Operation::AcceptEdit { proposal_id } => {
                let proposal = self
                    .proposals
                    .get(proposal_id)
                    .expect("snapshot validated")
                    .clone();
                let entry = self
                    .streams
                    .get_mut(&actor)
                    .expect("actor exists")
                    .working
                    .get_mut(&proposal.entry_id)
                    .expect("entry exists");
                let current = entry.current();
                if current.version != proposal.expected_version {
                    return Err("edit conflict: stale working-entry version".into());
                }
                let version = current
                    .version
                    .checked_add(1)
                    .ok_or("working-entry version overflow")?;
                entry.versions.push(EntryVersion {
                    version,
                    content: proposal.replacement,
                });
                self.proposals
                    .get_mut(proposal_id)
                    .expect("proposal exists")
                    .status = ProposalStatus::Accepted;
                Ok(format!(
                    "accepted edit {proposal_id}; entry version {version}"
                ))
            }
            Operation::RejectEdit { proposal_id } => {
                self.proposals
                    .get_mut(proposal_id)
                    .expect("snapshot validated")
                    .status = ProposalStatus::Rejected;
                Ok(format!("rejected edit {proposal_id}"))
            }
            Operation::MemoryQuery { query } => {
                let entries = self.memory.query(
                    &self.config.scope,
                    query,
                    self.config.max_retrieved_entries,
                    self.config.max_retrieved_bytes,
                );
                let count = entries.len();
                self.streams
                    .get_mut(&actor)
                    .expect("actor exists")
                    .retrieved = entries;
                Ok(format!("retrieved {count} memory entries"))
            }
            Operation::MemoryPropose {
                content,
                kind,
                source_references,
            } => {
                self.check_pending_budget()?;
                let draft = self.memory_draft(actor, content, *kind, source_references)?;
                // Validate against other staged changes too, avoiding a known-bad
                // queue that could block every task-boundary commit.
                let mut trial = self.staged_memory()?;
                trial.commit(draft.clone())?;
                self.pending_memory.push(PendingMemory::New(draft));
                Ok("memory staged for task-boundary commit".into())
            }
            Operation::MemoryRevise {
                entry_id,
                expected_version,
                content,
                kind,
                source_references,
            } => {
                self.check_pending_budget()?;
                let draft = self.memory_draft(actor, content, *kind, source_references)?;
                let mut trial = self.staged_memory()?;
                trial.revise(*entry_id, *expected_version, draft.clone())?;
                self.pending_memory.push(PendingMemory::Revise {
                    id: *entry_id,
                    version: *expected_version,
                    draft,
                });
                Ok("memory revision staged for task-boundary commit".into())
            }
            Operation::Finish { result } => {
                self.streams.get_mut(&actor).expect("actor exists").result = Some(result.clone());
                Ok("stream finished".into())
            }
        }
    }

    fn staged_memory(&self) -> Result<MemoryStore, String> {
        let mut trial = self.memory.clone();
        for pending in &self.pending_memory {
            match pending {
                PendingMemory::New(draft) => {
                    trial.commit(draft.clone())?;
                }
                PendingMemory::Revise { id, version, draft } => {
                    trial.revise(*id, *version, draft.clone())?;
                }
            }
        }
        Ok(trial)
    }

    fn check_pending_budget(&self) -> Result<(), String> {
        if self.pending_memory.len() >= self.config.max_pending_memory {
            return Err("pending memory budget exhausted".into());
        }
        Ok(())
    }

    fn record_outcome(&mut self, round: u64, outcome: &ActionOutcome) {
        let stream = self.streams.get_mut(&outcome.stream).expect("known actor");
        // Only a bounded feedback window enters the model; the full audit stays
        // available separately and is never rewritten by peer edits.
        stream.notices.push(format!(
            "{}: {}",
            if outcome.accepted { "ok" } else { "error" },
            outcome.detail
        ));
        if stream.notices.len() > 8 {
            stream.notices.remove(0);
        }
        self.trace.push(TraceEvent::Action {
            round,
            outcome: outcome.clone(),
        });
    }
}

#[cfg(test)]
#[path = "state_performance_tests.rs"]
mod performance_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::new(EngineConfig::default(), MemoryStore::new(8192)).unwrap()
    }
    fn publish(content: &str) -> Operation {
        Operation::Publish {
            target: None,
            content: content.into(),
        }
    }

    #[test]
    fn edits_require_owner_acceptance_and_preserve_history() {
        let mut e = engine();
        let entry = e.seed_entry(1, "hypothesis").unwrap();
        e.step_actions(vec![(
            0,
            Operation::ProposeEdit {
                target: 1,
                entry_id: entry,
                expected_version: 1,
                replacement: "corrected".into(),
            },
        )])
        .unwrap();
        assert_eq!(
            e.streams[&1].working[&entry].current().content,
            "hypothesis"
        );
        assert!(
            !e.step_actions(vec![(0, Operation::AcceptEdit { proposal_id: 1 })])
                .unwrap()
                .outcomes[0]
                .accepted
        );
        assert!(
            e.step_actions(vec![(1, Operation::AcceptEdit { proposal_id: 1 })])
                .unwrap()
                .outcomes[0]
                .accepted
        );
        assert_eq!(e.streams[&1].working[&entry].versions.len(), 2);
        assert_eq!(e.streams[&1].working[&entry].current().content, "corrected");
        assert_eq!(
            e.streams[&1].working[&entry].versions[0].content,
            "hypothesis"
        );
        assert_eq!(e.trace.len(), 3);
    }

    #[test]
    fn same_round_proposal_cannot_be_accepted() {
        let mut e = engine();
        let entry = e.seed_entry(1, "old").unwrap();
        let r = e
            .step_actions(vec![
                (1, Operation::AcceptEdit { proposal_id: 1 }),
                (
                    0,
                    Operation::ProposeEdit {
                        target: 1,
                        entry_id: entry,
                        expected_version: 1,
                        replacement: "new".into(),
                    },
                ),
            ])
            .unwrap();
        assert!(r.outcomes[0].accepted);
        assert!(!r.outcomes[1].accepted);
        assert_eq!(e.streams[&1].working[&entry].current().version, 1);
    }

    #[test]
    fn stale_edit_is_rejected_without_erasing_proposal() {
        let mut e = engine();
        let entry = e.seed_entry(1, "old").unwrap();
        for replacement in ["first", "second"] {
            e.step_actions(vec![(
                0,
                Operation::ProposeEdit {
                    target: 1,
                    entry_id: entry,
                    expected_version: 1,
                    replacement: replacement.into(),
                },
            )])
            .unwrap();
        }
        e.step_actions(vec![(1, Operation::AcceptEdit { proposal_id: 1 })])
            .unwrap();
        let r = e
            .step_actions(vec![(1, Operation::AcceptEdit { proposal_id: 2 })])
            .unwrap();
        assert!(!r.outcomes[0].accepted);
        assert_eq!(e.streams[&1].working[&entry].current().content, "first");
        assert_eq!(e.proposals[&2].status, ProposalStatus::Pending);
    }

    #[test]
    fn tokens_execute_only_when_complete_and_do_not_rewrite_history() {
        let mut e = engine();
        let frame = publish("hello").frame();
        for byte in &frame[..frame.len() - 1] {
            e.step_tokens(vec![(0, *byte)]).unwrap();
        }
        assert!(e.workspace.is_empty());
        e.step_tokens(vec![(0, b'\n')]).unwrap();
        assert_eq!(e.workspace[0].content, "hello");
        assert_eq!(e.streams[&0].history, frame);
    }

    #[test]
    fn malformed_utf8_and_unknown_policy_fields_do_not_execute() {
        let mut e = engine();
        for frame in [
            vec![255, b'\n'],
            b"{\"op\":\"finish\",\"result\":\"x\",\"scope\":\"other\"}\n".to_vec(),
        ] {
            for byte in frame {
                e.step_tokens(vec![(0, byte)]).unwrap();
            }
        }
        assert!(!e.streams[&0].is_finished());
        assert_eq!(
            e.trace
                .iter()
                .filter(
                    |event| matches!(event, TraceEvent::Action { outcome, .. } if !outcome.accepted)
                )
                .count(),
            2
        );
    }

    #[test]
    fn oversized_frame_suffix_cannot_execute() {
        let mut e = Engine::new(
            EngineConfig {
                max_frame_bytes: 10,
                ..EngineConfig::default()
            },
            MemoryStore::new(8192),
        )
        .unwrap();
        let bytes = [vec![b'x'; 12], publish("injected").frame()].concat();
        for byte in bytes {
            e.step_tokens(vec![(0, byte)]).unwrap();
        }
        assert!(e.workspace.is_empty());
        assert!(e.streams[&0].frame.is_empty());
    }

    #[test]
    fn invalid_round_is_transactional() {
        let mut e = engine();
        assert!(e.step_tokens(vec![(0, b'a'), (0, b'b')]).is_err());
        assert_eq!(e.round(), 0);
        assert!(e.streams[&0].history.is_empty());
        assert!(e.step_actions(vec![(100, publish("hello"))]).is_err());
        assert_eq!(e.round(), 0);
    }

    #[test]
    fn memory_is_scoped_and_not_visible_before_task_boundary() {
        let mut e = engine();
        let op = Operation::MemoryPropose {
            content: "verified answer is 42".into(),
            kind: MemoryKind::Fact,
            source_references: vec!["test fixture".into()],
        };
        assert!(e.step_actions(vec![(0, op)]).unwrap().outcomes[0].accepted);
        assert!(e.memory.active("default").is_empty());
        assert!(e.commit_memory().is_err());
        e.step_actions(vec![
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
        ])
        .unwrap();
        let entries = e.commit_memory().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].scope, "default");
        assert!(e.commit_memory().unwrap().is_empty());
    }

    #[test]
    fn memory_budget_rejects_excess_staging_without_poisoning_queue() {
        let mut e = Engine::new(EngineConfig::default(), MemoryStore::new(5)).unwrap();
        let op = || Operation::MemoryPropose {
            content: "four".into(),
            kind: MemoryKind::Fact,
            source_references: vec!["source".into()],
        };
        assert!(e.step_actions(vec![(0, op())]).unwrap().outcomes[0].accepted);
        assert!(!e.step_actions(vec![(1, op())]).unwrap().outcomes[0].accepted);
        e.step_actions(vec![
            (
                0,
                Operation::Finish {
                    result: "ok".into(),
                },
            ),
            (
                1,
                Operation::Finish {
                    result: "ok".into(),
                },
            ),
        ])
        .unwrap();
        assert_eq!(e.commit_memory().unwrap().len(), 1);
    }
}
