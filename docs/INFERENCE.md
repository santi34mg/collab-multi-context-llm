# Local GGUF inference and teacher distillation

The repository can load the installed **Qwen3.5-4B-Q5_K_M** and **LFM2.5-8B-A1B-Q4_K_M** GGUF models directly. LFM is the 8B-total/approximately-1B-active MoE model, not a dense 2.5B model.

There are two inference paths:

1. **Native:** Rust owns the inference request lifecycle and calls llama.cpp's C library in process. No `llama-cli` subprocess or running teacher server is required.
2. **HTTP debugging:** Rust calls an already-running `qwen-serve` or `lfm-serve` through its local OpenAI-compatible API.

These are frozen teacher models. The research transformer and its editable multi-context state are separate: loading a GGUF does not convert it into the proposed collaborative architecture or make its weights trainable here.

## Native setup

The optional `native` feature uses the **exact installed llama.cpp b11669 / commit `620d2225c` ABI**. Headers and the upstream MIT license are vendored; implementation libraries and weights are not copied into the repo. See [the ABI notes](../vendor/llama.cpp/README.md) before changing versions.

For this machine's verified installation:

```sh
export LLAMA_CPP_DIR=/home/santisil/opt/local-models/llama-b11669-cuda12/llama-b11669
export LD_LIBRARY_PATH=/home/santisil/opt/local-models/llama-b11669-cuda12/cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64
cargo build --release --features native
```

The C bridge requires Linux, a system C compiler, and `ar`. `LLAMA_CPP_DIR` is supplied at build time and its absolute library path is embedded in the binary. Another machine must provide compatible libraries and rebuild. The tested CPU backend plugins must be available alongside the libraries. CUDA execution requires working drivers and runtime dependencies; it was **not** validated here. CPU inference was validated for both actual models.

Without `native`, all ordinary student training and the local HTTP client still build with just Rust, `serde`, and `serde_json`.

## llama.cpp-style model execution

```sh
# The aliases resolve the existing files under HOME/.local/share/local-models.
./target/release/agent-collab infer --preset qwen -p 'What is 2 + 2? Reply with only the digit.' -n 64 --temp 0 --json
./target/release/agent-collab infer --preset lfm -p 'What is 2 + 2? Reply with only the digit.' -n 256 --temp 0

# Or select any compatible GGUF explicitly.
./target/release/agent-collab infer -m /path/to/model.gguf -p 'Explain persistent memory in one sentence.' -n 256

# Raw completion bypasses chat formatting and final-answer extraction.
./target/release/agent-collab infer -m /path/to/model.gguf --raw -p 'The capital of France is' -n 32
```

Common options:

| Option | Meaning |
| --- | --- |
| `-m`, `--model` | Local GGUF file; exclusive with `--preset` and `--endpoint`. |
| `--preset qwen\|lfm` | Existing local Qwen or LFM file; never downloads weights. |
| `-p`, `--prompt` / `-f`, `--file` | Prompt text or a UTF-8 prompt file. |
| `--system` | Optional system message in chat mode. |
| `-n`, `--max-tokens`, `--n-predict` | Maximum generated teacher tokens; default 256. |
| `-c`, `--ctx-size` | Native context capacity; default 2048. |
| `-b`, `--batch-size` | Prefill batch size; default 128, capped by context size. |
| `-t`, `--threads` | Native CPU threads; default 2. |
| `-ngl`, `--n-gpu-layers` | Offloaded layers; default 0, negative requests all layers. |
| `--temp`, `--top-p`, `--seed` | Sampling settings; temperature 0 is greedy. |
| `--stop TEXT` | Repeatable generation stop strings. |
| `--thinking` | Enable Qwen's verified thinking branch; final answer is still extracted. |
| `--raw` | Unformatted native completion, with generated text preserved. |
| `--json` | Print the completion text and token counts as JSON. |

Chat mode is the default. This CLI implements a useful subset, **not all llama.cpp flags or API features**. Each request resets both attention and hybrid recurrent state, uses the model's tokenizer, and checks that prompt plus requested output fits the context. There is no automatic context shifting.

### Reasoning handling

For the installed Qwen GGUF, native chat mode uses its verified empty-thinking template suffix by default. The installed LFM template has no equivalent generation-time switch: its reasoning still consumes tokens. Native chat mode extracts the final answer after one completed leading reasoning block and errors if reasoning is unfinished or ambiguous.

For example, a native LFM request for the single digit `4` used 90 tokens in a verified test; a 64-token limit failed safely rather than returning an unfinished thought as an answer. More complex questions can require considerably more. Token usage includes reasoning even though returned final text does not.

Unknown GGUF templates do not receive guessed model-specific formatting. The backend supports the pinned C API's template set, not an arbitrary Jinja interpreter. Raw mode remains available where appropriate.

## Run this repository's inference server

```sh
./target/release/agent-collab serve --preset qwen --alias qwen3.5-4b --port 8082
```

The Rust service owns the loaded native model and exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

It serves one request at a time. It is **loopback-only, unauthenticated, non-streaming, and intended for trusted local research**, not production or exposure through a public proxy. Request bodies are bounded; unsupported streaming requests are rejected explicitly. `--max-requests N` is available for bounded test runs.

## Use the existing debug helpers

Both existing helpers default to port 8080. Run one at a time, or override their ports in separate terminals:

```sh
qwen-serve --port 8080
lfm-serve --port 8081
```

Loading both consumes substantially more RAM/VRAM. For CPU-only debugging, append `--n-gpu-layers 0 --ctx-size 2048 --threads 2`; the helpers otherwise request a large context and GPU offloading.

```sh
cargo run --release -- infer --endpoint http://127.0.0.1:8080 -p 'What is 2 + 2? Reply with only the digit.'
cargo run --release -- infer --endpoint http://127.0.0.1:8081 --teacher-id lfm2.5-8b-a1b -p 'Explain context windows in one sentence.' -n 512
```

The client discovers model IDs through `/v1/models`; `--teacher-id` optionally selects an advertised ID. It ignores separate `reasoning_content` fields and does not substitute them for missing answers. HTTP access is restricted to loopback without TLS; no prompts leave the machine through this client.

## Collect a knowledge-distillation dataset

Prompts are a JSON array of `{id, messages, expected_answer?}`. See [the example prompts](../examples/distillation-prompts.json). `expected_answer`, where provided, must match exactly after trimming whitespace.

### Direct native teachers

Run sequentially so only one large teacher is loaded at a time:

```sh
./target/release/agent-collab distill --preset qwen --input examples/distillation-prompts.json --output target/qwen.jsonl -n 256 --ctx-size 1024 --temp 0
./target/release/agent-collab distill --preset lfm --input examples/distillation-prompts.json --output target/lfm.jsonl -n 512 --ctx-size 1024 --temp 0
```

### Existing HTTP teachers

```sh
cargo run --release -- distill --endpoint http://127.0.0.1:8080 --input examples/distillation-prompts.json --output target/qwen-http.jsonl -n 256 --temp 0
cargo run --release -- distill --endpoint http://127.0.0.1:8081 --input examples/distillation-prompts.json --output target/lfm-http.jsonl -n 512 --temp 0
```

Every JSONL sample records its source model, original messages, completed response, token counts, sampling settings, prompt ID, and format version. Different teachers can answer the same prompt ID. Dataset writes never overwrite an existing destination, and collection failures do not publish a partial dataset.

Blank, length-truncated, reasoning-marked, and explicitly incorrect answers are rejected. An unchecked teacher answer is **not automatically factual ground truth**: use task-specific validators and held-out evaluation before scaling collection. Raw reasoning text is not used as the student answer.

This is **sequence-level distillation**: supervise the teachers' final text using the student's byte vocabulary. Teacher token IDs, hidden states, and logits are not copied into the student; their tokenizers and architecture differ. No gradients flow into either quantized GGUF model.

## Train and reuse the student

```sh
cargo run --release -- train-distill --input target/qwen.jsonl --input target/lfm.jsonl --epochs 10 --width 32 --ctx-size 512 --streams 2 --output target/student.json

# Continue from an existing student instead of randomly initializing it.
cargo run --release -- train-distill --input target/qwen.jsonl --epochs 2 --resume target/student.json --output target/student-next.json

# Then warm-start the existing multi-context operation trainer.
cargo run --release -- train-trace examples/collaboration-dataset.json 10 target/collaborative-student.json target/student.json
```

Distillation applies loss **only to assistant response bytes**, using causal context and cycling through non-neutral stream identities. Long examples slide older context out of the configured window, but every response byte remains a training target. Reported initial/final losses are measured on the training samples, not held-out capability scores. Resuming training preserves checkpoint dimensions; do not supply new dimension flags.

A separate bounded student-text decoder uses the same role-labelled prompt format:

```sh
cargo run --release -- student --checkpoint target/student.json -p 'What is 2 + 2?' --max-bytes 16
```

The tiny student has no learned EOS for this text format; it emits a fixed maximum byte count and errors on invalid UTF-8. The teacher has its own proper EOG-based decoding. A handful of teacher answers cannot transfer billions of parameters' worth of capability into a width-32 model. Larger student architectures, training corpora, and evaluations remain necessary. **RL is not implemented by this distillation pipeline.**

## Validation

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --features native
cargo clippy --features native --all-targets -- -D warnings

# Opt in: starts each helper, verifies an answer, then always stops the process.
cargo test --release --test local_teachers -- --ignored --nocapture
```

The native ABI guide also documents ignored live tests for both GGUF files. Ordinary tests include mock HTTP transports and an end-to-end collect → train student → warm-start collaboration workflow.

Actual verified native collection produced all three example answers from both models. Combining those six samples for two small-student training epochs reduced in-sample byte loss from **5.5491 to 3.2411**. This validates execution and gradient flow, not generalization or improved collaborative reasoning.
