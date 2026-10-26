//! Opt-in, bounded integration against the user's installed helper commands.
//! No background process survives the test, including on assertion failure.

use agent_collab::distillation::{self, DistillPrompt};
use agent_collab::inference::http::HttpBackend;
use agent_collab::inference::{ChatMessage, GenerationConfig};
use std::fs::{self, File};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server {
    child: Child,
    log: PathBuf,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_file(&self.log);
    }
}

#[test]
#[ignore = "loads both local GGUF models sequentially; requires qwen-serve and lfm-serve"]
fn distills_verified_answers_from_installed_helpers() {
    for (command, model) in [("qwen-serve", "qwen3.5-4b"), ("lfm-serve", "lfm2.5-8b-a1b")] {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let log =
            std::env::temp_dir().join(format!("agent-collab-{command}-{}.log", std::process::id()));
        let child = Command::new(command)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--ctx-size",
                "1024",
                "--threads",
                "2",
                "--n-gpu-layers",
                "0",
                "--parallel",
                "1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(File::create(&log).unwrap()))
            .spawn()
            .unwrap();
        let mut server = Server { child, log };
        let url = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(90);
        let mut backend = loop {
            if let Some(status) = server.child.try_wait().unwrap() {
                panic!(
                    "{command} exited with {status}: {}",
                    fs::read_to_string(&server.log).unwrap_or_default()
                );
            }
            match HttpBackend::connect(&url, Some(model.into()), Duration::from_secs(2)) {
                Ok(backend) => break backend,
                Err(error) => {
                    assert!(
                        Instant::now() < deadline,
                        "{command} did not become ready: {error}; {}",
                        fs::read_to_string(&server.log).unwrap_or_default()
                    );
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        };
        // Generation can be longer than the readiness probe, especially on CPU.
        backend = HttpBackend::connect(&url, Some(model.into()), Duration::from_secs(90))
            .unwrap_or(backend);
        let options = GenerationConfig {
            max_tokens: 256,
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        let prompts = [DistillPrompt {
            id: "verified-addition".into(),
            messages: vec![ChatMessage::user(
                "What is 2 + 2? Reply with only the digit.",
            )],
            expected_answer: Some("4".into()),
        }];
        let samples =
            distillation::collect(&mut backend, &prompts, &options).unwrap_or_else(|error| {
                panic!(
                    "{command}: {error}; {}",
                    fs::read_to_string(&server.log).unwrap_or_default()
                )
            });
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].response.trim(), "4");
        assert_eq!(samples[0].teacher, model);
        assert_eq!(samples[0].finish_reason, "stop");
        println!(
            "{command}: verified final answer {:?}, {} teacher tokens",
            samples[0].response, samples[0].completion_tokens
        );
    }
}
