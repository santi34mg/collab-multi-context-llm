# collab-multi-context-llm

A Rust research prototype for language models that work across multiple cooperating context windows.

Instead of coordinating separate chat models, the project explores collaboration inside one shared-weight model. Each generation stream has its own working context, can publish selected information to peers, propose edits to another stream's context, and save selected results as memory for later sessions.

The repository includes a small trainable transformer, an engine that enforces the collaboration rules, and optional local GGUF teachers for inference and student training. It is an experimental reference implementation, not a pretrained assistant.

## Try it

Requires Rust with edition 2024 support. The built-in demo and tests need no model downloads or GPU.

```text
cargo run --release -- demo
```

The scripted demo walks through sharing information, accepting a peer edit, and saving and retrieving a result from `memory.json`.

## Documentation

- [Usage guide](docs/USAGE.md) — run the demos, train from demonstrations, and use checkpoints.
- [Inference and distillation](docs/INFERENCE.md) — run local GGUF teachers and train a student from their answers.
- [Implementation reference](docs/IMPLEMENTATION.md) — model internals, collaboration protocol, memory, tests, and current limitations.
- [Design proposal](DESIGN.md) — the research goals and proposed architecture.
- [Performance audit](docs/performance/REPORT.md) — benchmarks, measured optimizations, and reproduction instructions.
