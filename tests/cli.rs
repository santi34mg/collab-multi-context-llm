use agent_collab::memory::MemoryStore;
use agent_collab::model::TinyTransformer;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "agent-collab-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_agent-collab"))
}

#[test]
fn help_and_invalid_arguments_have_correct_status() {
    let help = command().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("train-trace"));
    assert!(!command().arg("unknown").output().unwrap().status.success());
    assert!(
        !command()
            .args(["train", "0"])
            .output()
            .unwrap()
            .status
            .success()
    );
}

#[test]
fn demo_executes_protocol_and_reloads_durable_memory() {
    let dir = TestDir::new();
    let memory = dir.0.join("memory.json");
    let trace = dir.0.join("trace.json");
    let output = command()
        .arg("demo")
        .arg(&memory)
        .arg(&trace)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("recalled memory 1 v1"));
    let store = MemoryStore::load(&memory).unwrap();
    assert_eq!(store.active("demo").len(), 1);
    assert!(store.active("other").is_empty());
    let events: Vec<Value> = serde_json::from_slice(&fs::read(trace).unwrap()).unwrap();
    assert!(events.iter().any(|event| event["event"] == "memory_commit"));
    assert!(
        events
            .iter()
            .any(|event| event["outcome"]["detail"] == "accepted edit 1; entry version 2")
    );
}

#[test]
fn inference_emits_parallel_tokens_and_does_not_save_unfinished_memory() {
    let dir = TestDir::new();
    let checkpoint = dir.0.join("model.json");
    let episode = dir.0.join("episode.json");
    let memory = dir.0.join("memory.json");
    TinyTransformer::new(agent_collab::experiments::small_config(), 7)
        .unwrap()
        .save(&checkpoint)
        .unwrap();
    fs::write(&episode, include_str!("../examples/finish-episode.json")).unwrap();
    let output = command()
        .arg("run")
        .arg(checkpoint)
        .arg(episode)
        .arg("1")
        .arg(&memory)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("task unfinished"));
    let events: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["round"], 0);
    assert_eq!(events[1]["round"], 0);
    assert_eq!(events[0]["stream"], 0);
    assert_eq!(events[1]["stream"], 1);
    assert!(!memory.exists());
}

#[test]
fn example_dataset_trains_and_produces_loadable_checkpoint() {
    let dir = TestDir::new();
    let dataset = dir.0.join("dataset.json");
    let checkpoint = dir.0.join("checkpoint.json");
    fs::write(
        &dataset,
        include_str!("../examples/collaboration-dataset.json"),
    )
    .unwrap();
    let output = command()
        .arg("train-trace")
        .arg(dataset)
        .arg("1")
        .arg(&checkpoint)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("425 supervised bytes"));
    assert_eq!(
        TinyTransformer::load(checkpoint)
            .unwrap()
            .config()
            .max_streams,
        3
    );
}
