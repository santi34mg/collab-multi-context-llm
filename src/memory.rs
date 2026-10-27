//! Versioned textual memory with deterministic, exact-scope lexical retrieval.
//!
//! Mutations are in-memory until [`MemoryStore::save`] succeeds. The budget counts
//! UTF-8 content bytes in *all* retained versions, including retracted entries;
//! metadata and serialized JSON overhead are not charged. Retraction changes the
//! latest version's status without appending a duplicate version. Only active
//! entries can be revised or retracted.
//!
//! Persistence assumes a single writer per destination path, including the check
//! of an existing file and its replacement. It is not a concurrent-writer locking
//! protocol. Saves stage an exclusively created file in the same directory, sync
//! its contents, then atomically rename it; on Unix the directory is also synced.
//! A directory-sync error can therefore be reported after replacement occurred.
//! An interrupted save may leave an unused temporary file; load never recovers
//! from temporary files or silently repairs invalid data.
//!
//! `forget` removes every version from this store. Saving back to the same path
//! replaces that file without retaining a backup, but cannot erase other store
//! instances, previously saved paths, crash leftovers, filesystem snapshots, or
//! backups, and does not provide secure erasure of underlying storage.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// Keep the persisted format identifier stable across project renames.
const FILE_FORMAT: &str = "agent-collab-memory";
const FILE_VERSION: u32 = 1;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Fact,
    Decision,
    Procedure,
    Preference,
    Hypothesis,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Superseded,
    Retracted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEntry {
    pub id: u64,
    pub version: u64,
    pub content: String,
    pub kind: MemoryKind,
    pub scope: String,
    pub source_references: Vec<String>,
    pub status: MemoryStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryDraft {
    pub content: String,
    pub kind: MemoryKind,
    pub scope: String,
    pub source_references: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MemoryStore {
    max_bytes: usize,
    retained_bytes: usize,
    next_id: u64,
    entries: BTreeMap<u64, Vec<MemoryEntry>>,
}

// A flat sequence lets validation detect duplicate IDs/versions instead of
// allowing a deserializer to silently overwrite duplicate map keys.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    format: String,
    format_version: u32,
    max_bytes: usize,
    next_id: u64,
    entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            retained_bytes: 0,
            next_id: 1,
            entries: BTreeMap::new(),
        }
    }

    pub fn commit(&mut self, draft: MemoryDraft) -> Result<MemoryEntry, String> {
        validate_fields(&draft.content, &draft.scope, &draft.source_references)?;
        let retained_bytes = self.check_budget(draft.content.len())?;
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| "memory ID space exhausted".to_string())?;
        let entry = entry_from_draft(self.next_id, 1, draft);
        self.entries.insert(entry.id, vec![entry.clone()]);
        self.next_id = next_id;
        self.retained_bytes = retained_bytes;
        Ok(entry)
    }

    /// Append a new active version; earlier content and provenance are retained.
    /// Scope is immutable, and a failed edit leaves the store unchanged.
    pub fn revise(
        &mut self,
        id: u64,
        expected_version: u64,
        draft: MemoryDraft,
    ) -> Result<MemoryEntry, String> {
        let current = self.editable(id, expected_version)?;
        validate_fields(&draft.content, &draft.scope, &draft.source_references)?;
        if current.scope != draft.scope {
            return Err("a memory revision cannot change scope".into());
        }
        let version = current
            .version
            .checked_add(1)
            .ok_or_else(|| "memory version space exhausted".to_string())?;
        let retained_bytes = self.check_budget(draft.content.len())?;
        let entry = entry_from_draft(id, version, draft);
        let history = self.entries.get_mut(&id).expect("validated memory ID");
        history.last_mut().expect("nonempty history").status = MemoryStatus::Superseded;
        history.push(entry.clone());
        self.retained_bytes = retained_bytes;
        Ok(entry)
    }

    /// Mark the latest version retracted, retaining its version and byte charge.
    pub fn retract(&mut self, id: u64, expected_version: u64) -> Result<(), String> {
        self.editable(id, expected_version)?;
        self.entries
            .get_mut(&id)
            .expect("validated memory ID")
            .last_mut()
            .expect("nonempty history")
            .status = MemoryStatus::Retracted;
        Ok(())
    }

    /// Delete all versions and release their budget. IDs are never reused.
    /// See the module documentation for the limits of physical deletion.
    pub fn forget(&mut self, id: u64) -> Result<(), String> {
        let history = self
            .entries
            .remove(&id)
            .ok_or_else(|| format!("unknown memory ID {id}"))?;
        for entry in history {
            self.retained_bytes -= entry.content.len();
        }
        Ok(())
    }

    /// Rank active entries in the exact scope by distinct word overlap, greatest
    /// first, breaking ties by ascending ID. Words are Unicode alphanumeric runs
    /// lowercased without stemming; repeated words do not boost relevance.
    /// Oversized entries are skipped so smaller matches can fill the result.
    /// The result budget, like the store budget, counts only content bytes.
    pub fn query(
        &self,
        scope: &str,
        query: &str,
        max_entries: usize,
        max_bytes: usize,
    ) -> Vec<MemoryEntry> {
        let query_words: Vec<_> = words(query).into_iter().collect();
        if query_words.is_empty() || max_entries == 0 || max_bytes == 0 {
            return Vec::new();
        }
        // Track distinct matches by entry ID instead of building an owned word
        // set for every entry. Normalize the entire string before splitting to
        // preserve Unicode lowercasing and word-boundary semantics.
        let mut matched = vec![None; query_words.len()];
        let mut ranked: Vec<_> = self
            .active_entries(scope)
            .filter_map(|entry| {
                let normalized = entry.content.to_lowercase();
                let mut score = 0;
                for word in normalized.split(|character: char| !character.is_alphanumeric()) {
                    if let Ok(index) =
                        query_words.binary_search_by(|query| query.as_str().cmp(word))
                        && matched[index] != Some(entry.id)
                    {
                        matched[index] = Some(entry.id);
                        score += 1;
                    }
                }
                (score > 0).then_some((score, entry))
            })
            .collect();
        ranked.sort_unstable_by(|(score_a, a), (score_b, b)| {
            score_b.cmp(score_a).then_with(|| a.id.cmp(&b.id))
        });
        let mut remaining = max_bytes;
        let mut result = Vec::new();
        for (_, entry) in ranked {
            if entry.content.len() <= remaining {
                remaining -= entry.content.len();
                result.push(entry.clone());
                if result.len() == max_entries {
                    break;
                }
            }
        }
        result
    }

    /// Active latest versions in the exact scope, ordered by ascending ID.
    pub fn active(&self, scope: &str) -> Vec<MemoryEntry> {
        self.active_entries(scope).cloned().collect()
    }

    /// Retained history in ascending version order, or empty for an unknown ID.
    pub fn versions(&self, id: u64) -> Vec<MemoryEntry> {
        self.entries.get(&id).cloned().unwrap_or_default()
    }

    /// Atomically save under the single-writer contract in the module docs.
    /// Existing destinations must be regular files containing a valid compatible
    /// store; arbitrary, corrupt, or unsupported files are never overwritten.
    /// Parent directories must already exist. No backup is retained.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if path.file_name().is_none() {
            return Err("memory save path must name a file".into());
        }
        match fs::symlink_metadata(path) {
            Ok(metadata) => {
                if !metadata.file_type().is_file() {
                    return Err("refusing to replace a non-regular memory file".into());
                }
                Self::load(path).map_err(|error| {
                    format!("refusing to replace incompatible memory file: {error}")
                })?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("cannot inspect memory destination: {error}")),
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let snapshot = Snapshot {
            format: FILE_FORMAT.into(),
            format_version: FILE_VERSION,
            max_bytes: self.max_bytes,
            next_id: self.next_id,
            entries: self.entries.values().flatten().cloned().collect(),
        };
        let (mut temporary, mut file) = create_temporary(parent, path)?;
        let write_result = (|| {
            serde_json::to_writer_pretty(&mut file, &snapshot)
                .map_err(|error| format!("cannot serialize memory: {error}"))?;
            file.write_all(b"\n")
                .map_err(|error| format!("cannot write memory: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("cannot sync memory: {error}"))
        })();
        // Close before rename/cleanup, including on platforms that lock open files.
        drop(file);
        write_result?;
        fs::rename(&temporary.path, path)
            .map_err(|error| format!("cannot replace memory file: {error}"))?;
        temporary.remove_on_drop = false;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("memory replaced but directory sync failed: {error}"))?;
        Ok(())
    }

    /// Load only a complete supported snapshot, validating all history invariants.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref())
            .map_err(|error| format!("cannot open memory file: {error}"))?;
        let snapshot: Snapshot = serde_json::from_reader(BufReader::new(file))
            .map_err(|error| format!("invalid memory JSON: {error}"))?;
        Self::from_snapshot(snapshot)
    }

    fn from_snapshot(snapshot: Snapshot) -> Result<Self, String> {
        if snapshot.format != FILE_FORMAT || snapshot.format_version != FILE_VERSION {
            return Err("unsupported memory file format or version".into());
        }
        if snapshot.next_id == 0 {
            return Err("invalid next memory ID".into());
        }
        let mut store = Self::new(snapshot.max_bytes);
        store.next_id = snapshot.next_id;
        let mut previous_id = 0;
        for entry in snapshot.entries {
            validate_fields(&entry.content, &entry.scope, &entry.source_references)?;
            if entry.id == 0 || entry.id >= store.next_id || entry.id < previous_id {
                return Err("memory IDs must be positive, ordered, and below next_id".into());
            }
            let retained_bytes = store.check_budget(entry.content.len())?;
            let history = store.entries.entry(entry.id).or_default();
            let expected_version = match history.last() {
                None => 1,
                Some(previous) => {
                    if previous.status != MemoryStatus::Superseded {
                        return Err(
                            "only the latest memory version may be active or retracted".into()
                        );
                    }
                    if previous.scope != entry.scope {
                        return Err("memory history changes scope".into());
                    }
                    previous
                        .version
                        .checked_add(1)
                        .ok_or_else(|| "memory version overflow".to_string())?
                }
            };
            if entry.version != expected_version {
                return Err("memory versions must start at 1 and increase contiguously".into());
            }
            previous_id = entry.id;
            history.push(entry);
            store.retained_bytes = retained_bytes;
        }
        if store.entries.values().any(|history| {
            history.last().expect("nonempty history").status == MemoryStatus::Superseded
        }) {
            return Err("latest memory version cannot be superseded".into());
        }
        Ok(store)
    }

    fn check_budget(&self, additional: usize) -> Result<usize, String> {
        self.retained_bytes
            .checked_add(additional)
            .filter(|total| *total <= self.max_bytes)
            .ok_or_else(|| "retained memory content exceeds byte budget".to_string())
    }

    fn editable(&self, id: u64, expected_version: u64) -> Result<&MemoryEntry, String> {
        let current = self
            .entries
            .get(&id)
            .and_then(|history| history.last())
            .ok_or_else(|| format!("unknown memory ID {id}"))?;
        if current.version != expected_version {
            return Err(format!(
                "stale memory version for ID {id}: expected {expected_version}, current {}",
                current.version
            ));
        }
        if current.status != MemoryStatus::Active {
            return Err(format!("memory ID {id} is not active"));
        }
        Ok(current)
    }

    fn active_entries<'a>(&'a self, scope: &'a str) -> impl Iterator<Item = &'a MemoryEntry> {
        self.entries
            .values()
            .filter_map(|history| history.last())
            .filter(move |entry| entry.status == MemoryStatus::Active && entry.scope == scope)
    }
}

fn validate_fields(content: &str, scope: &str, references: &[String]) -> Result<(), String> {
    if content.trim().is_empty() {
        return Err("memory content must not be empty".into());
    }
    if scope.trim().is_empty() {
        return Err("memory scope must not be empty".into());
    }
    if references.is_empty()
        || references
            .iter()
            .any(|reference| reference.trim().is_empty())
    {
        return Err("memory provenance requires nonempty source references".into());
    }
    Ok(())
}

fn entry_from_draft(id: u64, version: u64, draft: MemoryDraft) -> MemoryEntry {
    MemoryEntry {
        id,
        version,
        content: draft.content,
        kind: draft.kind,
        scope: draft.scope,
        source_references: draft.source_references,
        status: MemoryStatus::Active,
    }
}

fn words(text: &str) -> BTreeSet<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

struct TemporaryPath {
    path: PathBuf,
    remove_on_drop: bool,
}

impl Drop for TemporaryPath {
    fn drop(&mut self) {
        if self.remove_on_drop {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn create_temporary(parent: &Path, destination: &Path) -> Result<(TemporaryPath, File), String> {
    for _ in 0..128 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".collab-multi-context-llm-memory-{}-{sequence}.tmp",
            std::process::id()
        ));
        if path.file_name() == destination.file_name() {
            continue;
        }
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                return Ok((
                    TemporaryPath {
                        path,
                        remove_on_drop: true,
                    },
                    file,
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot create temporary memory file: {error}")),
        }
    }
    Err("cannot allocate a unique temporary memory file".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            for _ in 0..128 {
                let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "collab-multi-context-llm-memory-tests-{}-{sequence}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("cannot create test directory: {error}"),
                }
            }
            panic!("cannot allocate test directory");
        }

        fn file(&self) -> PathBuf {
            self.0.join("memory.json")
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn draft(content: &str, scope: &str) -> MemoryDraft {
        MemoryDraft {
            content: content.into(),
            kind: MemoryKind::Fact,
            scope: scope.into(),
            source_references: vec!["observation:1".into()],
        }
    }

    fn ids(entries: Vec<MemoryEntry>) -> Vec<u64> {
        entries.into_iter().map(|entry| entry.id).collect()
    }

    #[test]
    fn saved_memory_remains_compatible_after_project_rename() {
        let snapshot: Snapshot = serde_json::from_str(
            r#"{
            "format": "agent-collab-memory", "format_version": 1,
            "max_bytes": 128, "next_id": 2,
            "entries": [{
                "id": 1, "version": 1, "content": "kept", "kind": "fact",
                "scope": "scope", "source_references": ["observation:1"],
                "status": "active"
            }]
        }"#,
        )
        .unwrap();
        let store = MemoryStore::from_snapshot(snapshot).unwrap();
        assert_eq!(store.active("scope")[0].content, "kept");
    }

    #[test]
    fn retrieval_matches_owned_set_reference_for_unicode_and_budgets() {
        let mut store = MemoryStore::new(8192);
        for content in [
            "Alpha ALPHA beta",
            "Café 東京",
            "ΟΣ ος ΟΣΑ",
            "İ I i",
            "unmatched",
            "beta",
        ] {
            store.commit(draft(content, "scope")).unwrap();
            store.commit(draft(content, "other")).unwrap();
        }
        store
            .revise(1, 1, draft("beta beta Café", "scope"))
            .unwrap();
        store.retract(3, 1).unwrap();
        store.forget(5).unwrap();
        for query in [
            "",
            "!!!",
            "ALPHA beta beta",
            "café 東京",
            "ΟΣ ΟΣΑ",
            "İ i",
            "unmatched",
        ] {
            for limit in [0, 1, 3, 20] {
                for budget in [0, 4, 16, 8192] {
                    let query_words = words(query);
                    let mut ranked: Vec<_> = store
                        .active_entries("scope")
                        .filter_map(|entry| {
                            let score = words(&entry.content).intersection(&query_words).count();
                            (score > 0).then_some((score, entry))
                        })
                        .collect();
                    ranked.sort_unstable_by(|(a_score, a), (b_score, b)| {
                        b_score.cmp(a_score).then_with(|| a.id.cmp(&b.id))
                    });
                    let mut remaining = budget;
                    let mut expected = Vec::new();
                    if limit > 0 && budget > 0 {
                        for (_, entry) in ranked {
                            if entry.content.len() <= remaining {
                                remaining -= entry.content.len();
                                expected.push(entry.clone());
                                if expected.len() == limit {
                                    break;
                                }
                            }
                        }
                    }
                    assert_eq!(store.query("scope", query, limit, budget), expected);
                }
            }
        }
    }

    #[test]
    fn serde_uses_snake_case_and_roundtrips_public_types() {
        for (kind, name) in [
            (MemoryKind::Fact, "fact"),
            (MemoryKind::Decision, "decision"),
            (MemoryKind::Procedure, "procedure"),
            (MemoryKind::Preference, "preference"),
            (MemoryKind::Hypothesis, "hypothesis"),
        ] {
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(name));
            assert_eq!(
                serde_json::from_value::<MemoryKind>(json!(name)).unwrap(),
                kind
            );
        }
        for (status, name) in [
            (MemoryStatus::Active, "active"),
            (MemoryStatus::Superseded, "superseded"),
            (MemoryStatus::Retracted, "retracted"),
        ] {
            assert_eq!(serde_json::to_value(status).unwrap(), json!(name));
        }
        let draft = draft("a fact", "project");
        assert_eq!(
            serde_json::from_value::<MemoryDraft>(serde_json::to_value(&draft).unwrap()).unwrap(),
            draft
        );
        let entry = MemoryStore::new(100).commit(draft).unwrap();
        assert_eq!(
            serde_json::from_value::<MemoryEntry>(serde_json::to_value(&entry).unwrap()).unwrap(),
            entry
        );
    }

    #[test]
    fn validates_drafts_without_consuming_ids_or_mutating_history() {
        let mut store = MemoryStore::new(100);
        let mut invalid = vec![draft(" \n\t", "project"), draft("fact", " \t")];
        let mut missing = draft("fact", "project");
        missing.source_references.clear();
        invalid.push(missing);
        let mut blank = draft("fact", "project");
        blank.source_references.push(" \n".into());
        invalid.push(blank);
        for candidate in &invalid {
            assert!(store.commit(candidate.clone()).is_err());
        }
        let entry = store.commit(draft("valid", "project")).unwrap();
        assert_eq!((entry.id, entry.version), (1, 1));
        for candidate in invalid {
            assert!(store.revise(entry.id, 1, candidate).is_err());
            assert_eq!(store.versions(entry.id), vec![entry.clone()]);
        }
        assert!(MemoryStore::new(0).commit(draft("x", "project")).is_err());
    }

    #[test]
    fn stale_edits_and_retracted_edits_are_rejected() {
        let mut store = MemoryStore::new(100);
        let first = store.commit(draft("old", "project")).unwrap();
        let second = store.revise(first.id, 1, draft("new", "project")).unwrap();
        assert_eq!(second.version, 2);
        assert!(
            store
                .revise(first.id, 1, draft("stale", "project"))
                .is_err()
        );
        assert!(store.retract(first.id, 1).is_err());
        assert_eq!(store.active("project"), vec![second.clone()]);
        store.retract(first.id, 2).unwrap();
        let history = store.versions(first.id);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].status, MemoryStatus::Superseded);
        assert_eq!(history[0].content, "old");
        assert_eq!(history[1].status, MemoryStatus::Retracted);
        assert_eq!(history[1].version, 2);
        assert!(store.active("project").is_empty());
        assert!(store.query("project", "new", 10, 100).is_empty());
        assert!(
            store
                .revise(first.id, 2, draft("revive", "project"))
                .is_err()
        );
        assert!(store.retract(first.id, 2).is_err());
        assert_eq!(store.versions(first.id), history);
        assert!(store.revise(99, 1, draft("x", "project")).is_err());
        assert!(store.retract(99, 1).is_err());
        assert!(store.versions(99).is_empty());
    }

    #[test]
    fn retrieval_is_scoped_ranked_and_deterministic() {
        let mut store = MemoryStore::new(1000);
        let a = store.commit(draft("Rust rust RUST", "project")).unwrap();
        let b = store.commit(draft("safe Rust", "project")).unwrap();
        let c = store.commit(draft("rust", "project")).unwrap();
        store.commit(draft("rust safe", "project/child")).unwrap();
        store.commit(draft("rust safe", "other")).unwrap();
        store.commit(draft("unrelated", "project")).unwrap();
        let old = store.commit(draft("rust safe", "project")).unwrap();
        let revised = store
            .revise(old.id, 1, draft("replacement", "project"))
            .unwrap();
        let retracted = store.commit(draft("rust safe", "project")).unwrap();
        store.retract(retracted.id, 1).unwrap();
        for _ in 0..3 {
            assert_eq!(
                ids(store.query("project", "SAFE, RuSt! rust", 10, 1000)),
                vec![b.id, a.id, c.id]
            );
        }
        assert_eq!(ids(store.query("project", "rust", 1, 100)), vec![a.id]);
        assert_eq!(
            ids(store.active("project")),
            vec![a.id, b.id, c.id, 6, revised.id]
        );
        assert!(store.query("Project", "rust", 10, 1000).is_empty());
        assert!(store.query("project", "trust", 10, 1000).is_empty());
        assert!(store.query("project", " -- ", 10, 1000).is_empty());
        assert!(store.query("project", "rust", 0, 1000).is_empty());
        assert!(store.query("project", "rust", 10, 0).is_empty());
        let before = store.versions(a.id);
        assert!(store.revise(a.id, 1, draft("rust", "other")).is_err());
        assert_eq!(store.versions(a.id), before);
    }

    #[test]
    fn retrieval_skips_oversized_matches_and_counts_utf8_bytes() {
        let mut store = MemoryStore::new(100);
        store.commit(draft("rust safe enormous", "s")).unwrap();
        let short = store.commit(draft("rust", "s")).unwrap();
        let unicode = store.commit(draft("ÉTÉ", "s")).unwrap();
        assert_eq!(ids(store.query("s", "rust safe", 10, 4)), vec![short.id]);
        assert!(store.query("s", "été", 10, 4).is_empty());
        assert_eq!(store.query("s", "été", 10, 5), vec![unicode]);
        let mut exact = MemoryStore::new(2);
        exact.commit(draft("é", "s")).unwrap();
        assert!(exact.commit(draft("a", "s")).is_err());
    }

    #[test]
    fn revisions_charge_all_retained_content_and_fail_atomically() {
        let mut store = MemoryStore::new(6);
        let original = store.commit(draft("old", "s")).unwrap();
        let revised = store.revise(original.id, 1, draft("new", "s")).unwrap();
        let before = store.versions(original.id);
        assert!(store.revise(original.id, 2, draft("x", "s")).is_err());
        assert!(store.commit(draft("x", "s")).is_err());
        assert_eq!(store.versions(original.id), before);
        assert_eq!(store.active("s"), vec![revised]);
        store.retract(original.id, 2).unwrap();
        assert!(store.commit(draft("x", "s")).is_err());
        store.forget(original.id).unwrap();
        assert!(store.versions(original.id).is_empty());
        assert_eq!(store.commit(draft("abcdef", "s")).unwrap().id, 2);
        assert!(store.forget(original.id).is_err());
    }

    #[test]
    fn persistence_roundtrip_preserves_history_budget_and_id_watermark() {
        let directory = TestDirectory::new();
        let mut store = MemoryStore::new(15);
        let a = store.commit(draft("old", "s")).unwrap();
        store.revise(a.id, 1, draft("new", "s")).unwrap();
        store.retract(a.id, 2).unwrap();
        let b = store.commit(draft("live", "other")).unwrap();
        let forgotten = store.commit(draft("gone", "s")).unwrap();
        store.forget(forgotten.id).unwrap();
        store.save(directory.file()).unwrap();
        let bytes = fs::read(directory.file()).unwrap();
        let mut loaded = MemoryStore::load(directory.file()).unwrap();
        assert_eq!(loaded.versions(a.id), store.versions(a.id));
        assert_eq!(loaded.active("other"), vec![b]);
        assert!(loaded.versions(forgotten.id).is_empty());
        loaded.save(directory.file()).unwrap();
        assert_eq!(fs::read(directory.file()).unwrap(), bytes);
        assert!(loaded.commit(draft("123456", "s")).is_err());
        assert_eq!(loaded.commit(draft("12345", "s")).unwrap().id, 4);
        loaded.save(directory.file()).unwrap();
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
        assert_eq!(
            MemoryStore::load(directory.file()).unwrap().retained_bytes,
            15
        );
    }

    #[test]
    fn forgetting_removes_every_version_from_the_replaced_file() {
        let directory = TestDirectory::new();
        let mut store = MemoryStore::new(100);
        let entry = store.commit(draft("secret_original", "s")).unwrap();
        store
            .revise(entry.id, 1, draft("secret_revision", "s"))
            .unwrap();
        store.save(directory.file()).unwrap();
        store.forget(entry.id).unwrap();
        store.save(directory.file()).unwrap();
        let text = fs::read_to_string(directory.file()).unwrap();
        assert!(!text.contains("secret_original"));
        assert!(!text.contains("secret_revision"));
        let mut loaded = MemoryStore::load(directory.file()).unwrap();
        assert!(loaded.versions(entry.id).is_empty());
        assert_eq!(loaded.commit(draft("fresh", "s")).unwrap().id, 2);
    }

    #[test]
    fn temporary_files_are_distinct_exclusive_and_cleaned_up() {
        let directory = TestDirectory::new();
        let sequence = TEMP_SEQUENCE.load(Ordering::Relaxed);
        let destination = directory.0.join(format!(
            ".collab-multi-context-llm-memory-{}-{sequence}.tmp",
            std::process::id()
        ));
        let (temporary, file) = create_temporary(&directory.0, &destination).unwrap();
        assert_ne!(temporary.path, destination);
        assert_eq!(temporary.path.parent(), Some(directory.0.as_path()));
        assert!(!destination.exists());
        assert_eq!(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary.path)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::AlreadyExists
        );
        drop(file);
        drop(temporary);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);

        let sequence = TEMP_SEQUENCE.load(Ordering::Relaxed);
        let existing = directory.0.join(format!(
            ".collab-multi-context-llm-memory-{}-{sequence}.tmp",
            std::process::id()
        ));
        fs::write(&existing, b"leftover content").unwrap();
        let (temporary, file) = create_temporary(&directory.0, &directory.file()).unwrap();
        assert_ne!(temporary.path, existing);
        assert_eq!(fs::read(&existing).unwrap(), b"leftover content");
        drop(file);
        drop(temporary);
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    }

    #[test]
    fn corrupt_or_incompatible_destinations_are_not_replaced() {
        let directory = TestDirectory::new();
        let store = MemoryStore::new(100);
        for bytes in [
            b"not JSON".as_slice(),
            b"{}",
            b"{\"format\":\"another-format\",\"format_version\":1,\"max_bytes\":0,\"next_id\":1,\"entries\":[]}",
            b"{\"format\":\"agent-collab-memory\",\"format_version\":2,\"max_bytes\":0,\"next_id\":1,\"entries\":[]}",
        ] {
            fs::write(directory.file(), bytes).unwrap();
            assert!(MemoryStore::load(directory.file()).is_err());
            assert!(store.save(directory.file()).is_err());
            assert_eq!(fs::read(directory.file()).unwrap(), bytes);
            assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
        }
        assert!(store.save(&directory.0).is_err());
        assert!(MemoryStore::load(directory.0.join("missing")).is_err());
        assert!(store.save(directory.0.join("missing/parent.json")).is_err());
    }

    #[test]
    fn load_rejects_corrupt_history_invariants() {
        let directory = TestDirectory::new();
        let mut store = MemoryStore::new(100);
        let first = store.commit(draft("old", "s")).unwrap();
        store.revise(first.id, 1, draft("new", "s")).unwrap();
        store.commit(draft("another", "s")).unwrap();
        store.save(directory.file()).unwrap();
        let valid: Value = serde_json::from_slice(&fs::read(directory.file()).unwrap()).unwrap();
        let mut invalid = Vec::new();
        for (field, value) in [
            ("next_id", json!(0)),
            ("next_id", json!(2)),
            ("max_bytes", json!(8)),
            ("format_version", json!(999)),
            ("format", json!("other")),
            ("unknown_field", json!(true)),
        ] {
            let mut candidate = valid.clone();
            candidate[field] = value;
            invalid.push(candidate);
        }
        for (index, field, value) in [
            (0, "id", json!(0)),
            (0, "version", json!(0)),
            (0, "version", json!(2)),
            (1, "version", json!(1)),
            (1, "version", json!(3)),
            (0, "status", json!("active")),
            (0, "status", json!("retracted")),
            (1, "status", json!("superseded")),
            (1, "scope", json!("other")),
            (0, "content", json!(" \n")),
            (0, "scope", json!("")),
            (0, "source_references", json!([])),
            (0, "source_references", json!([""])),
            (0, "kind", json!("unknown")),
            (0, "unknown_field", json!(true)),
        ] {
            let mut candidate = valid.clone();
            candidate["entries"][index][field] = value;
            invalid.push(candidate);
        }
        let mut duplicate = valid.clone();
        let repeated = duplicate["entries"][1].clone();
        duplicate["entries"]
            .as_array_mut()
            .unwrap()
            .insert(2, repeated);
        invalid.push(duplicate);
        let mut unordered = valid.clone();
        unordered["entries"].as_array_mut().unwrap().swap(0, 2);
        invalid.push(unordered);
        for (index, candidate) in invalid.into_iter().enumerate() {
            let bytes = serde_json::to_vec(&candidate).unwrap();
            fs::write(directory.file(), &bytes).unwrap();
            assert!(
                MemoryStore::load(directory.file()).is_err(),
                "case {index}: {candidate}"
            );
            assert!(store.save(directory.file()).is_err(), "case {index}");
            assert_eq!(fs::read(directory.file()).unwrap(), bytes);
        }
        let valid_bytes = serde_json::to_vec(&valid).unwrap();
        for bytes in [
            valid_bytes[..valid_bytes.len() - 1].to_vec(),
            [valid_bytes.as_slice(), b" trailing"].concat(),
            b"{\"format\":\"a\",\"format\":\"b\"}".to_vec(),
        ] {
            fs::write(directory.file(), bytes).unwrap();
            assert!(MemoryStore::load(directory.file()).is_err());
        }
    }

    #[test]
    fn id_and_version_overflow_fail_without_mutation() {
        let mut store = MemoryStore::new(100);
        store.next_id = u64::MAX;
        assert!(store.commit(draft("x", "s")).is_err());
        assert!(store.entries.is_empty());
        assert_eq!(store.retained_bytes, 0);
        store.next_id = 1;
        let entry = store.commit(draft("old", "s")).unwrap();
        store.entries.get_mut(&entry.id).unwrap()[0].version = u64::MAX;
        assert!(store.revise(entry.id, u64::MAX, draft("new", "s")).is_err());
        assert_eq!(store.versions(entry.id)[0].status, MemoryStatus::Active);
        assert_eq!(store.retained_bytes, 3);
        store.retained_bytes = usize::MAX;
        assert!(store.check_budget(1).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn saving_refuses_symlinks_without_touching_their_targets() {
        let directory = TestDirectory::new();
        let store = MemoryStore::new(0);
        store.save(directory.file()).unwrap();
        let link = directory.0.join("link.json");
        std::os::unix::fs::symlink(directory.file(), &link).unwrap();
        let before = fs::read(directory.file()).unwrap();
        assert!(store.save(link).is_err());
        assert_eq!(fs::read(directory.file()).unwrap(), before);
    }
}
