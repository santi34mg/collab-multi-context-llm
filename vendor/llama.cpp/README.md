# Pinned llama.cpp native ABI

These public headers and `LICENSE` are from **ggml-org/llama.cpp commit
`620d2225c`, build b11669, version `0.7.0-dev`** (shared library version
`libllama.so.0.7.0`; ggml shared library version `0.27.0`). Sources:

- <https://github.com/ggml-org/llama.cpp/tree/620d2225c>
- `include/llama.h`
- `ggml/include/ggml.h`, `ggml-cpu.h`, `ggml-backend.h`, `ggml-alloc.h`,
  `ggml-opt.h`, and `gguf.h`

The last three headers supply the transitive includes needed by `llama.h`.
Their raw source URLs are
`https://raw.githubusercontent.com/ggml-org/llama.cpp/620d2225c/ggml/include/<name>`.
The upstream MIT license is preserved in `LICENSE`.

## What is linked

This directory does **not** contain a replacement transformer implementation or
compiled llama.cpp. The optional `native` feature links an external, compatible
llama.cpp installation **in process**. That library supplies GGUF parsing,
architecture-specific execution (including Qwen3.5 hybrid recurrence and LFM2.5
MoE/convolution state), tokenization, chat-template recognition, sampling, and
CPU/GPU kernel backends. There is no HTTP server, CLI subprocess, or homemade
mapping of these models onto the project's separate trainable transformer.

`native/bridge.c` is compiled with the system `cc` and archived with `ar` by
`build.rs`, using only `std::process::Command` (no build dependencies). All llama
parameter structs are constructed and passed by value **only in C**. Rust sees
opaque handles, scalar values, borrowed byte/token buffers, and pointer arrays.
`NativeModel` owns the model/context/sampler and releases them with RAII.

**The exact commit matters:** llama.cpp's by-value parameter structs are not a
stable ABI, and another build with the same `0.7.0-dev` version string can still
be incompatible. Do not substitute another revision's shared libraries or
backend plugins without updating and testing all headers together. Initialization
checks the exported version string, but that string cannot verify the commit;
this pin is an installation requirement, not a runtime commit attestation.

## Build and runtime setup (Linux)

Without `--features native`, `cargo test` requires neither llama.cpp nor a C
compiler. With the feature, **`LLAMA_CPP_DIR` must be explicitly set at build
time** to a directory containing `libllama.so`, `libggml.so`, and
`libggml-base.so` (or a prefix with these in `lib/` or `lib64/`). Unversioned
linker symlinks must exist. Compatible `libggml-cpu-*.so` plugins must be in that
same library directory; ordinary installations may instead provide
`libggml-cpu.so`. The compatible system C++ runtime and OpenMP runtime must also
be installed. `CC` and `AR`, if specified, name executable paths, not shell
commands with additional arguments.

The build canonicalizes the library directory and embeds an **absolute rpath**.
The directory is also passed explicitly to `ggml_backend_load_all_from_path`,
followed by `ggml_backend_load_all`, once per process. Initialization verifies
that a CPU device was actually registered. Backends remain loaded for process
lifetime so dropping one model cannot invalidate another. Relocating the
installation requires rebuilding. Runtime loader overrides such as
`LD_LIBRARY_PATH` must not select an incompatible llama/ggml installation.

For the installed b11669 build on this machine:

```sh
export LLAMA_CPP_DIR=/home/santisil/opt/local-models/llama-b11669-cuda12/llama-b11669
export LD_LIBRARY_PATH=/home/santisil/opt/local-models/llama-b11669-cuda12/cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64
cargo build --features native
```

These paths are examples of the tested installation, **not hardcoded build
paths**. CUDA runtime dependencies can be supplied with `LD_LIBRARY_PATH` or the
system dynamic loader configuration. The bridge does not guess a CUDA SDK or
runtime directory. The bundled plugin loader can attempt to initialize CUDA
even for CPU-only inference and print a no-device/driver diagnostic in a
sandbox. With `gpu_layers: 0`, the model uses an empty accelerator device list
and disables context operation/KV offloading, so CPU inference still works.
GPU execution itself requires accessible drivers and was not tested in the
sandbox.

## Rust API and request semantics

```rust,ignore
use std::path::PathBuf;
use collab_multi_context_llm::inference::{ChatMessage, GenerationConfig, InferenceBackend};
use collab_multi_context_llm::inference::native::{NativeConfig, NativeModel};

let mut model = NativeModel::load(NativeConfig {
    model: PathBuf::from("/path/to/model.gguf"),
    context_size: 512,
    batch_size: 32,
    threads: 2,
    gpu_layers: 0,
})?;
let options = GenerationConfig {
    max_tokens: 128,
    temperature: 0.0,
    ..GenerationConfig::default()
};
let chat = model.generate(&[ChatMessage::user("Say hello.")], &options)?;
let raw = model.complete("The capital of France is", &options)?;
```

- `NativeModel::load(NativeConfig) -> Result<Self, String>` requires positive
  context size and thread count, with `1 <= batch_size <= context_size` and
  context size at most `i32::MAX`. Negative `gpu_layers` requests all layers,
  per this llama.cpp revision. Each model owns one mutable, non-shared context.
- `apply_chat_template` is unchanged: it exposes the model's own
  `llama_model_chat_template` through `llama_chat_apply_template`, including the
  assistant prefix. This upstream API recognizes a supported template set,
  **not arbitrary Jinja**. Missing/unrecognized templates return an error, never
  a guessed fallback. `InferenceBackend::generate` validates messages, applies
  the thinking policy below, and returns final-answer text only.
- `complete(&mut self, prompt: &str, options: &GenerationConfig)` is the raw
  completion path (appropriate for CLI `-p`). It does not wrap the prompt in a
  chat template. Both paths use the model's own tokenizer with configured
  special-token insertion and special-token recognition enabled.
- Prompts are prefilled in bounded batches. Actual context/batch capacities
  are queried after context creation, capped by the requested limits.
  `prompt_tokens + max_tokens` must fit the context; excess requests fail
  explicitly instead of silently truncating or context-shifting.
- Temperature zero is greedy. Otherwise sampling applies temperature, top-p,
  and seeded distribution sampling, with a new sampler for each request.
  Upstream reserves seed `u32::MAX` for a random seed. Fixed seeds are intended
  to reproduce results on the same model/backend/configuration, not across
  different devices or library builds.
- All EOG tokens terminate generation. Stop strings match accumulated bytes,
  including matches across token boundaries; the earliest matched stop is
  excluded from returned text. UTF-8 conversion happens **once after byte
  aggregation**, not per token. Invalid output bytes or an incomplete final
  codepoint become replacement characters in the final Rust string.
- `Generation` reports the model path as `model`, actual tokenizer prompt
  count, and the number of sampled completion tokens, including a terminating
  EOG or tokens that contributed an excluded stop string. `finish_reason` is
  `stop` for EOG/stop-string termination and `length` for the token limit.
- `llama_memory_clear(memory, true)` resets attention **and recurrent state**
  before execution and after every completed/failed execution. Requests never
  inherit another request's hybrid state. Errors release partial model loads,
  and decode errors retain the owned model but clear its request state.

## Thinking and final-answer safety

`NativeModel::set_thinking(bool)` defaults to **false**. It affects only
`generate`; `apply_chat_template` retains its original upstream behavior and
`complete` remains an unfiltered raw completion API.

The pinned upstream source
[`src/llama-chat.cpp`](https://github.com/ggml-org/llama.cpp/blob/620d2225c/src/llama-chat.cpp)
contains **neither `qwen3` nor `qwen3-nothink`** in its builtin template map.
Its detector maps both installed models' embedded Jinja templates to basic
ChatML, so `apply_chat_template` ends with `<|im_start|>assistant\n` for both,
without executing either model's thinking-related Jinja logic.

Model-specific behavior is gated by **`general.architecture` metadata read via
`llama_model_meta_val_str` and verified embedded template source**, never by the
model filename:

- **Qwen3.5** (`qwen35`, installed `general.name=Qwen3.5-4B`): the actual embedded
  generation branch explicitly renders `<think>\n\n</think>\n\n` when
  `enable_thinking` is false/undefined, or `<think>\n` when true. After checking
  the exact source branch and C-rendered assistant suffix, `generate` appends
  precisely the chosen suffix. This reproduces that verified branch; it does
  not claim an unsupported builtin name or implement arbitrary Jinja.
- **LFM2.5** (`lfm2moe`, installed name
  `Feb5E04Da4910Dd56D33F0Cd03747Ff298C1E801`): its embedded generation branch
  contains **no thinking-disable switch**. `preserve_thinking` only controls
  historical assistant messages. `generate` leaves its actual prompt unchanged
  and safely extracts the final answer from a completed leading reasoning block.
  The thinking setting cannot suppress this model's internal reasoning. Empty
  assistant-thought prefills were tested and reopened reasoning; they are **not
  used** as a purported no-thinking workaround.

For either recognized reasoning protocol, `generate` removes **exactly one
leading** `<think>...</think>` block, or consumes the matching `</think>` when
Qwen's actual prompt already prefilled the opening tag. Tags later inside an
ordinary final answer are not stripped. Unfinished reasoning (including a token
limit or stop string before the closing delimiter), reasoning without a final
answer, repeated leading reasoning blocks, and ambiguous/partial delimiters
return an **error instead of hidden thoughts**. Leading reasoning from an
unrecognized model/template is also rejected instead of guessing its protocol.
Token counts still include all sampled reasoning/EOG tokens, and prompt counts
include any Qwen reasoning prefill. Final-answer truncation retains
`finish_reason=length` for the collector to reject normally.

### Actual CPU arithmetic results

Prompt: `What is 2 + 2? Reply with only the digit.`, greedy, context 512,
threads 2, GPU layers 0:

| Model | Token budget | Returned final text | Sampled tokens | Finish |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B-Q5_K_M | 64 | `4` | 2 | `stop` |
| LFM2.5-8B-A1B-Q4_K_M | 64 | Error: unfinished reasoning; no text returned | 64 before rejection | — |
| LFM2.5-8B-A1B-Q4_K_M | 128 | `4` | 90, including hidden reasoning | `stop` |

Thus this LFM build **does not meet a 64-token final-answer budget** for that
prompt. Use at least a sufficient budget (128 worked here), and handle errors
rather than collecting unfinished reasoning. No retry or budget expansion is
performed implicitly by `generate`.

## Validation against the actual GGUF files

Ordinary native unit tests do not load model weights. The ignored live tests
require explicit GGUF paths and use **context 512, threads 2, gpu_layers 0**,
serially, without starting a server. Raw regression requests use max tokens 8;
final-answer requests use 64, with an explicit 128-token LFM fallback in the test:

```sh
export LLAMA_CPP_DIR=/home/santisil/opt/local-models/llama-b11669-cuda12/llama-b11669
export LD_LIBRARY_PATH=/home/santisil/opt/local-models/llama-b11669-cuda12/cudart-llama-b11669-bin-ubuntu-cuda-12.8-x64
export NATIVE_TEST_QWEN_MODEL=/home/santisil/.local/share/local-models/Qwen3.5-4B-Q5_K_M.gguf
export NATIVE_TEST_LFM_MODEL=/home/santisil/.local/share/local-models/LFM2.5-8B-A1B-Q4_K_M.gguf
cargo test --features native --lib inference::native::tests -- --include-ignored --test-threads=1 --nocapture
```

Both model tests exercise the actual GGUF chat template and generated text,
multi-batch prefilling, raw completion, greedy repeatability after a different
request, seeded sampling repeatability, stop-string termination, and explicit
prompt/context/option bound failures. They additionally verify arithmetic final
answers, explicit thinking opt-in, unchanged original templates, and rejection
of unfinished reasoning. Small unit tests cover byte-based UTF-8/stop matching,
metadata/source-gated reasoning policy, safe leading-block extraction (including
already-prefilled reasoning), and configuration rejection before model loading.
