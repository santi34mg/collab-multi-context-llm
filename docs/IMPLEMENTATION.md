# Implementation reference

This document describes the current Rust prototype. The [design proposal](../DESIGN.md) describes the broader research direction; the [usage guide](USAGE.md) covers executable workflows.

## Module map

| Module | Responsibility |
| --- | --- |
| [`inference`](../src/inference.rs), [`native`](../src/inference/native.rs), [`http`](../src/inference/http.rs) | Shared teacher interface, in-process GGUF execution, local debugging client/server. |
| [`distillation`](../src/distillation.rs) | Validated teacher JSONL datasets and answer-only student training. |
| [`commands`](../src/commands.rs) | Inference, server, distillation, and student-text CLI. |
| [`model`](../src/model.rs) | Deterministic initialization, single-layer transformer, analytic backpropagation, SGD, checkpoints. |
| [`context`](../src/context.rs) | Structured token blocks, stream/type/position features, visibility masks, simultaneous query rows. |
| [`state`](../src/state.rs) | Snapshot rounds, byte framing, versioned working entries, edit proposals, validation, audit events. |
| [`memory`](../src/memory.rs) | Scoped lexical retrieval, version histories, retention budgets, retraction/deletion, atomic persistence. |
| [`training`](../src/training.rs) | Demonstration validation and multi-stream teacher forcing. |
| [`experiments`](../src/experiments.rs) | Exhaustive published-bit smoke task and communication ablation. |
| [`main`](../src/main.rs) | CLI demos, training, and bounded free-running inference. |

Default runtime dependencies are `serde` and `serde_json`; Criterion is a development dependency. The optional `native` feature adds the Linux llama.cpp bridge and requires compatible external libraries, a C compiler, and local weights. See the [native setup guide](INFERENCE.md#native-setup).

## Neural computation and context visibility

The default student has width 16, one attention head, one decoder layer, and a width-32 tanh feed-forward network. It uses residual connections, a 257-symbol vocabulary (256 bytes plus input-only BOS), and trainable token, position, stream, and context-kind embeddings. Every parameter group is trained by analytic backpropagation; finite-difference tests check gradients across all groups.

One model input contains query positions for every active stream. Attention selects authorized working, history, publication, inbox, and retrieved-memory tokens. Peer histories are private; only explicit publications are shared. Positions advance across entries within a reader/source/type window, preserving entry/content association. Publication source identity is structural rather than inferred from a prose envelope.

The encoder duplicates shared payloads per reader to preserve reader-local positions and prevent hidden-message lengths from affecting visible positions. Publications with no active readers are skipped. Dense masks and duplicated payloads remain scalability limits.

Only requested query rows are evaluated by the single layer. Other mask rows are identity rows. Stacking layers requires redesigning the masks and contextual representations, not repeating the current layer. State is re-encoded every round, so context edits cannot leave stale KV-cache entries.

Default CLI limits are 2,048 encoded tokens, up to 256 recent history bytes per stream, and 512 positions per reader/source/type window. Full generated history stays in the engine up to its separate byte limit. Non-history context overflow is an explicit error, not silent truncation. Limits and ablations are configurable through the library.

## Collaboration protocol

Operations are newline-delimited UTF-8 JSON:

```json
{"op":"publish","target":null,"content":"a = 17"}
{"op":"propose_edit","target":1,"entry_id":2,"expected_version":1,"replacement":"total = 42"}
{"op":"accept_edit","proposal_id":1}
{"op":"memory_propose","content":"17 + 25 = 42","kind":"fact","source_references":["checked arithmetic fixture"]}
{"op":"finish","result":"42"}
```

Also supported: `reject_edit`, `memory_query`, and `memory_revise`. The full typed schema is [`state::Operation`](../src/state.rs). Ordinary bytes accumulate locally; only complete valid frames execute. Oversized frames are discarded through the next newline. Non-ASCII characters require multiple decoding rounds because tokenization is byte-level.

### Round and edit rules

- Eligibility checks use the committed snapshot from the start of the round.
- A proposal cannot be accepted in the same round in which it is created.
- Only the target stream may accept or reject a peer edit.
- Accepted edits append a working-entry version; prior content and generation history remain available to the host.
- Stale versions fail explicitly. Valid operations commit in ascending stream-ID order.
- The application fixes stream count and memory scope; generated fields cannot change those policies.

Simultaneous output distributions do not imply multithreaded execution. See the [performance audit](performance/REPORT.md) for measured costs and independent-request scaling.

## Durable memory

`memory_propose` and `memory_revise` stage changes. Once all streams finish, `Engine::commit_memory` applies the batch atomically in RAM. `MemoryStore::save` makes it durable; the engine never silently writes to disk.

Entries retain kind, scope, provenance, ID, version, and status. Retrieval uses deterministic case-insensitive word overlap within the exact authorized scope, with entry and byte limits. Storage budgets count UTF-8 content bytes across all retained versions, including superseded and retracted versions. Metadata and JSON overhead are not charged.

Saving creates an exclusive same-directory temporary file, syncs it, and renames it over the destination. On Unix, the directory is synced too. The store assumes one writer per file; it is not a concurrent database. Corrupt or incompatible files are rejected rather than repaired or overwritten. Model checkpoints use a separate JSON format and ordinary non-atomic file writes.

The persisted format identifier remains `agent-collab-memory` for compatibility with files saved before the project rename; no data migration is required.

The memory library also exposes `revise`, `retract`, and `forget`. Forgetting removes all retained versions from this store, but cannot erase traces, other process copies, snapshots, or backups. It does not provide secure storage erasure. Scope and provenance validation are mechanical controls, not factual verification or a complete prompt-injection defense.

## Validation

```text
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release --test learning -- --ignored
```

Tests cover gradients, masked-token noninterference, learning, checkpoint roundtrips, snapshot visibility, stale edits, owner permissions, invalid byte frames, budgets, scopes, corruption, durable reload, deletion, and executable teacher forcing. Regression tests also compare optimized engine transitions with the original implementation, backward-kernel accumulation bits, and Unicode retrieval results.

Native-feature checks and opt-in tests requiring local teachers are documented in [Inference and distillation](INFERENCE.md#validation).

## Performance benchmarks

```text
cargo bench --bench performance
cargo run --release --example performance_results -- build optimized
cargo run --release --example performance_results -- run target/performance/optimized-bench local-run 0
cargo run --release --example performance_results -- describe docs/performance/data/local-run.json
```

The Criterion suite covers context encoding, prediction, SGD, complete in-memory training episodes, distillation, retrieval, cache-eviction stress, stream counts, and independent-request concurrency. Collection and reporting use compiled Rust utilities. Generated results stay local in the ignored `docs/performance/data/` directory.

The [performance report](performance/REPORT.md) contains the measurement environment, before/after comparisons, rejected experiments, profiling limitations, and full reproduction instructions. Student benchmarks do not measure external GGUF teachers or establish model capability.

## Implemented scope and limitations

The student is fully trainable but is not a pretrained LLM. The scripted demo exercises the protocol; the published-bit task tests learned use of peer information; trace training supervises operation generation. These do not establish autonomous collaboration, held-out generalization, or improved general reasoning.

Pretrained GGUF teachers are separate, frozen models. Their final answers can train the student through sequence-level distillation, not reinforcement learning or logit transfer. See [the inference guide](INFERENCE.md) for that workflow and its limits.

Research work still ahead includes:

- Held-out tasks, external message-passing and single-stream baselines, and multi-seed evaluation of quality, latency, and compute.
- Larger demonstration corpora and free-running operation/task-success evaluation.
- Outcome-based optimization of communication and memory utility.
- Learned retrieval/routing, dynamic stream counts, asynchronous scheduling, and latent memory.
- Sparse attention, larger tensor kernels, GPU execution, and cache-aware mutable context encoding.
- Stronger trust policies and externally verified memory commits.
