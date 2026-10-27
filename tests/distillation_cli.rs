use collab_multi_context_llm::inference::{
    ChatMessage, Generation, GenerationConfig, InferenceBackend,
};
use collab_multi_context_llm::{distillation, model::TinyTransformer};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "distill-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_collab-multi-context-llm"))
}
fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct Teacher;
impl InferenceBackend for Teacher {
    fn model_id(&self) -> &str {
        "qwen-debug"
    }
    fn generate(&mut self, _: &[ChatMessage], _: &GenerationConfig) -> Result<Generation, String> {
        Ok(Generation {
            text: "4".into(),
            model: self.model_id().into(),
            prompt_tokens: 8,
            completion_tokens: 2,
            finish_reason: "stop".into(),
        })
    }
}

#[test]
fn collection_training_and_collaboration_warm_start_work_end_to_end() {
    let dir = TestDir::new();
    let prompts = dir.0.join("prompts.json");
    let samples = dir.0.join("samples.jsonl");
    let student = dir.0.join("student.json");
    let dataset = dir.0.join("collaboration.json");
    let adapted = dir.0.join("adapted.json");
    fs::write(&prompts, r#"[{"id":"addition","messages":[{"role":"user","content":"What is 2+2?"}],"expected_answer":"4"}]"#).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        collab_multi_context_llm::inference::http::serve_listener(&mut Teacher, listener, Some(2))
            .unwrap()
    });
    let collected = command()
        .args(["distill", "--endpoint", &endpoint, "--input"])
        .arg(prompts)
        .arg("--output")
        .arg(&samples)
        .output()
        .unwrap();
    success(&collected);
    server.join().unwrap();
    let data = distillation::read_jsonl(&samples).unwrap();
    assert_eq!(data[0].teacher, "qwen-debug");
    assert_eq!(data[0].response, "4");
    let trained = command()
        .args(["train-distill", "--input"])
        .arg(&samples)
        .args([
            "--epochs",
            "2",
            "--width",
            "8",
            "--ctx-size",
            "128",
            "--output",
        ])
        .arg(&student)
        .output()
        .unwrap();
    success(&trained);
    assert_eq!(TinyTransformer::load(&student).unwrap().config().width, 8);
    let episode: serde_json::Value =
        serde_json::from_str(include_str!("../examples/finish-episode.json")).unwrap();
    fs::write(&dataset, serde_json::to_vec(&vec![episode]).unwrap()).unwrap();
    let fine_tuned = command()
        .arg("train-trace")
        .arg(dataset)
        .arg("1")
        .arg(&adapted)
        .arg(student)
        .output()
        .unwrap();
    success(&fine_tuned);
    assert_eq!(TinyTransformer::load(adapted).unwrap().config().width, 8); // did not reinitialize the default width-16 model
    // Refusing an existing dataset occurs before attempting a network connection.
    let protected = command()
        .args([
            "distill",
            "--endpoint",
            "http://127.0.0.1:1",
            "--input",
            "unused",
            "--output",
        ])
        .arg(&samples)
        .output()
        .unwrap();
    assert!(!protected.status.success());
    assert!(String::from_utf8_lossy(&protected.stderr).contains("already exists"));
    assert_eq!(distillation::read_jsonl(samples).unwrap().len(), 1);
}

#[cfg(not(feature = "native"))]
#[test]
fn missing_native_feature_has_actionable_error() {
    let output = command()
        .args(["infer", "--preset", "qwen", "-p", "hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features native"));
}

#[test]
fn command_help_and_invalid_options_do_not_load_models() {
    for name in ["infer", "serve", "distill", "train-distill", "student"] {
        success(&command().args([name, "--help"]).output().unwrap());
    }
    let invalid = command()
        .args(["infer", "--preset", "qwen", "-p", "hello", "--temp", "-1"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("temperature"));
}
