# Collaborative Multi-Context Language Model

**Status:** Research proposal with an executable Rust reference prototype; see the [implementation reference](docs/IMPLEMENTATION.md#implemented-scope-and-limitations) for implemented scope and limitations.

**Working name:** Collaborative Multi-Context Model (CMCM)

**Teacher bootstrapping:** Local Qwen3.5 and LFM2.5 GGUF inference, final-answer dataset collection, and student distillation are implemented; see [the inference guide](docs/INFERENCE.md). These frozen pretrained teachers remain separate from the experimental multi-context architecture. Outcome-based RL is not implemented.

## 1. Vision

Build a language model whose native unit of computation is a **collection of interacting context windows**, rather than a single token sequence.

Each active context represents a distinct stream of computation: an agent, a scratchpad, or a specialized workspace. Streams can generate concurrently, read selected information from one another, and propose changes to shared or peer-owned context. A subset of generated information can be committed to persistent memory contexts and reused in later sessions.

The goal is not to wrap independent chat models in an orchestration framework. Collaboration, context mutation, and memory access should be operations the model learns to perform as part of its computation.

This proposal is an architectural research direction, not a claim that these mechanisms will outperform existing models. The first implementation should make those claims testable at small scale.

## 2. Objectives

### 2.1 Native collaboration

- Maintain multiple independently evolving context windows within one model invocation.
- Produce up to one new token per active stream in a decoding round.
- Let streams selectively exchange information rather than broadcasting their entire histories.
- Let streams propose structured edits to other streams' working contexts.
- Learn when collaboration is useful, which information to share, and which edits to accept.

### 2.2 Persistent memory

- Let selected generated content survive beyond the current context or session.
- Store that content in addressable memory contexts, not only in external prose files.
- Learn what to preserve, retrieve, revise, and forget under explicit storage budgets.
- Distinguish temporary reasoning from durable knowledge, decisions, and user preferences.

### 2.3 Research quality

- Compare against strong single-stream and externally orchestrated baselines.
- Measure quality, total computation, latency, communication, and memory cost.
- Make collaboration and memory behavior inspectable and reproducible.

## 3. Non-goals for the initial prototype

- Training a frontier-scale foundation model from scratch.
- Demonstrating human-like consciousness, independent personalities, or general autonomous agency.
- Persisting every reasoning token indefinitely.
- Allowing unrestricted modification of instructions or arbitrary historical KV-cache entries.
- Treating persistent memory as online updates to model weights.
- Assuming that more streams or longer memory automatically improve performance.

## 4. Core concepts

| Concept | Meaning |
| --- | --- |
| Stream | A logical agent with a local generation history and working context. |
| Working context | Mutable information a stream can use during the current task. |
| Generation history | Append-only record of tokens and operations actually produced. |
| Shared workspace | A context accessible to multiple streams during a task. |
| Memory context | A durable, versioned collection of entries available across sessions. |
| Communication operation | A learned action that publishes, reads, or proposes an edit. |
| Decoding round | A synchronization boundary at which active streams generate outputs. |
| Commit | A validated change made visible to subsequent rounds or future sessions. |

An “agent” initially means a stream identity and its state, not a separate set of model weights. Streams share parameters; stream or role embeddings may differentiate their behavior.

## 5. Architectural principles

1. **Multiple contexts are first-class model inputs.** The model receives structured stream and memory state, rather than a flattened transcript alone.
2. **Communication is learned; execution is explicit.** The model chooses operations, while a small state engine applies deterministic visibility, permission, and conflict rules.
3. **History and editable state are different.** Editing a working context does not rewrite the audit record of what happened.
4. **Persistence is a controlled decision.** Temporary computation does not become durable by default.
5. **Information flow is budgeted.** Reading, sharing, and storing information consume measurable resources.
6. **Start with observable token representations.** Latent communication and latent memory are later research variants, not prerequisites.

A state engine is unavoidable for storage and scheduling. Its presence does not make this an orchestration-only design: the research distinction is that the model is jointly trained to use structured multi-context attention and collaboration operations, rather than coordinating unchanged chat models exclusively through prompts.

## 6. Proposed model state

At round `t`, the system state is:

```text
S_t = { H_i, W_i, Q_i } for each stream i
      + shared workspace B
      + persistent memory M
      + visibility and permission policy P
```

- `H_i`: append-only generation and operation history.
- `W_i`: versioned mutable working-context entries.
- `Q_i`: execution state, such as active, waiting, or finished.
- `B`: task-scoped shared entries and messages.
- `M`: persistent memory entries, including provenance and scope.
- `P`: allowed reads and writes, fixed by the application rather than editable by generated text.

The model computes a distribution over each active stream's next output given a readable projection of this state.

```text
p_theta(y_i,t | readable_state(S_t, i))
```

### Parallel generation semantics

All streams in round `t` read the same committed snapshot. Their outputs become visible in round `t + 1`, after validation and conflict resolution.

The simplest prototype factorizes the joint distribution within a round:

```text
p_theta(y_1,t, ..., y_n,t | S_t)
    = product_i p_theta(y_i,t | readable_state(S_t, i))
```

This does not model direct dependence between tokens sampled in the same round. Coordination happens through previous rounds. A more expressive coupled decoder is a later experiment.

Generating one token per stream is distinct from generating several sequential tokens ahead in a single stream. Batched execution may improve utilization, but does not guarantee lower total compute or latency.

## 7. Decoder architecture

### 7.1 Shared backbone, structured attention

Start with a small decoder-only transformer with shared weights across streams. Add:

- Stream identity and context-type embeddings.
- Local positions within each context; entry identifiers and versions for mutable state.
- An attention mask separating local history, readable peer state, shared state, and retrieved memory.
- Explicit operation tokens for collaboration and memory actions.

A stream may attend to its own causal history, its current working context, published peer entries, shared workspace entries, and authorized retrieved memory. It must not attend to future tokens or uncommitted outputs from the same round.

For the first implementation, represent selected contexts as token blocks with a block-structured attention mask. This is simple to inspect, although dense attention over all blocks can be expensive. Sparse routing or dedicated cross-attention modules can follow once usefulness is established.

### 7.2 Communication interface

Initial learned operations:

```text
PUBLISH(target, payload)
PROPOSE_EDIT(target, entry_id, expected_version, replacement)
ACCEPT_EDIT(proposal_id)
REJECT_EDIT(proposal_id)
MEMORY_QUERY(query, scope)
MEMORY_PROPOSE(content, scope, provenance)
MEMORY_REVISE(entry_id, expected_version, replacement)
FINISH(result)
```

Operations are serialized across ordinary decoding steps and take effect only when complete. Producing one token does not imply completing an entire edit in that round. Ordinary text remains local unless explicitly published.

For the prototype, target windows are fixed and memory retrieval is implemented by a deterministic retriever. Dynamic stream creation, learned routing, and learned retrieval can be introduced separately.

### 7.3 What counts as native collaboration?

The prototype should satisfy all of the following:

- One shared model processes structured multi-context state.
- Attention visibility is represented inside model computation.
- The model generates communication and edit operations.
- Training includes losses or rewards for successful collaboration.
- Evaluation includes ablations that remove these mechanisms.

Using multiple ordinary model calls with only text-message passing remains an important baseline, not the proposed endpoint.

## 8. Context mutation semantics

“Mutating another agent's context” means changing its editable working state, not silently rewriting its past outputs.

### 8.1 Initial policy: proposed edits

1. A source stream generates an edit proposal with a target entry and expected version.
2. The target stream can accept or reject it.
3. The state engine checks permissions and version validity.
4. An accepted edit creates a new entry version at the next commit boundary.
5. The old version and proposal remain in the audit log.

Start with entry-level replacement rather than arbitrary token-span edits. Entry identifiers are more stable than offsets in changing token sequences.

Direct writes to designated shared entries are a later controlled variant. System instructions, access policies, and the user's original input are never peer-editable.

### 8.2 Concurrent changes

Use optimistic version checks. A proposal based on a stale version fails explicitly and can be regenerated. If several proposals target the same version, use a deterministic ordering and report conflicts; do not silently merge incompatible text.

### 8.3 KV-cache correctness

A text edit invalidates cached representations that depended on the previous content. Merely replacing token IDs while retaining those cache entries is incorrect.

The reference implementation should recompute affected representations, even if this is slow. Later variants can use separately encoded mutable context blocks or dedicated cross-attention caches. Those variants still need explicit invalidation when source content changes.

## 9. Persistent memory contexts

### 9.1 Memory is durable state, not weights

A memory context survives process and session boundaries through storage. Its contents condition future inference without requiring gradient updates.

Persistence does not require loading every stored token on every forward pass. Use a durable backing store and a bounded active memory context populated by retrieval.

### 9.2 Initial memory representation

Use inspectable textual entries with structured metadata:

```text
MemoryEntry {
  id,
  version,
  content,
  kind: fact | decision | procedure | preference | hypothesis,
  scope,
  source_references,
  created_at,
  updated_at,
  status: active | superseded | retracted
}
```

Scopes separate users, projects, and tasks. Provenance distinguishes an observed fact from a model-generated hypothesis. A model's self-reported confidence, if added, is not evidence of correctness.

### 9.3 Memory lifecycle

1. **Propose:** A stream identifies useful content from current computation.
2. **Validate:** Check scope, provenance, duplication, permissions, and budget.
3. **Commit:** Persist an entry and its version history.
4. **Retrieve:** Select authorized entries relevant to a later task.
5. **Use:** Condition generation on a bounded memory context.
6. **Revise or retract:** Handle corrections without silently erasing provenance.
7. **Forget:** Apply retention limits and explicit deletion policies.

Initially, memory proposals commit at task boundaries. This prevents an unverified thought from immediately becoming durable evidence during the same task. Immediate writes can be evaluated later.

### 9.4 What should be retained?

Candidate durable content includes verified intermediate results, decisions and their rationale, reusable procedures, unresolved hypotheses clearly labeled as such, and explicit user preferences.

Raw internal reasoning should not be retained indiscriminately: it is often redundant, mistaken, sensitive, or useful only within one task. The model should learn to extract useful state from generated thinking. Selected original spans can be retained when appropriate; a later variant can test persistent latent representations.

### 9.5 Deletion and trust

Separate logical retraction from physical deletion. Privacy-sensitive deletion must remove content from active storage, retrieval indexes, and applicable retained copies according to an explicit policy.

Retrieved memory is data, not a higher-priority instruction. Durable entries must not acquire permission to override application policy merely because they were stored earlier.

## 10. Training strategy

### 10.1 Staged development

**Stage A — Reference execution environment**

Implement streams, snapshot rounds, structured operations, versioned state, and persistent storage. Use scripted policies to test semantics before training.

**Stage B — Supervised multi-context learning**

Train a small transformer on tasks with observable collaboration benefits. Generate synthetic traces with known dependencies, valid operations, and memory ground truth. Preserve simultaneous-round semantics in both inputs and labels.

**Stage C — Outcome-based optimization**

Optimize task success under communication, computation, and memory budgets. Compare supervised-only training with reinforcement learning or another outcome-based method. Do not assume one credit-assignment approach in advance.

**Stage D — Transfer and scaling**

Only after small-scale evidence, test pretrained-backbone adaptation, larger stream counts, learned routing, and longer-lived memory.

### 10.2 Candidate objective

A starting objective is:

```text
L = L_token
    + lambda_op * L_operation
    + lambda_mem * L_memory_selection
    + lambda_cost * L_resource_cost
```

- `L_token`: next-token loss summed over active streams.
- `L_operation`: validity and target-selection supervision, where separable labels exist.
- `L_memory_selection`: retain/retrieve/revise supervision on tasks with known relevant facts.
- `L_resource_cost`: a differentiable surrogate or outcome-based penalty for resource use.

Operation tokens already participate in token loss; auxiliary terms should only be added where they supply distinct supervision. Cost and delayed memory utility may require policy optimization rather than ordinary backpropagation through discrete actions.

### 10.3 Training-data requirements

- Include tasks that require communication and tasks best solved without it.
- Include disagreement, stale edits, false hypotheses, and correction opportunities.
- Split memory episodes by underlying facts and task generators to limit leakage.
- Ensure memory contains only information available before the evaluated query.
- Avoid training solely on a teacher's single preferred collaboration style.

## 11. Evaluation plan

### 11.1 Research hypotheses

| Hypothesis | Evidence required |
| --- | --- |
| Structured multi-context computation improves problem solving. | Better quality–compute tradeoff than matched single-stream and message-passing baselines. |
| Peer mutation helps beyond communication alone. | Gains over a read/publish-only variant, including edit-recovery tests. |
| Learned persistence improves future tasks. | Higher later-session accuracy with bounded storage and no evaluation leakage. |
| Selective communication is more efficient than full visibility. | Comparable quality with lower attention, communication, or latency cost. |
| Useful specialization emerges. | Distinct causal contributions, not just different-looking text. |

### 11.2 Initial task families

1. **Distributed evidence:** Each stream receives different clues needed for a joint answer.
2. **Collaborative constraint solving:** Streams derive partial assignments and repair conflicting assumptions.
3. **Draft and critique:** One stream proposes a solution while another identifies and proposes corrections.
4. **Episodic recall:** Relevant facts must survive unrelated intervening tasks and process restarts.
5. **Changing facts:** New evidence supersedes old memory; retrieval must prefer current information.
6. **Adversarial collaboration:** A stream or retrieved entry supplies misleading instructions or poisoned claims.

Synthetic tasks provide exact checks, but success on them is not evidence of general reasoning gains. Follow with held-out natural-language or programming tasks.

### 11.3 Baselines and ablations

- Single stream without persistence.
- Single stream with retrieval memory and the same storage budget.
- Independent parallel samples plus answer selection.
- Multiple ordinary agents using text messages.
- Multi-context model with publishing but no peer edits.
- Multi-context model with full visibility versus selective visibility.
- Fixed memory retention versus learned memory selection.
- Oracle communication or retrieval on synthetic tasks, as an upper-bound diagnostic.

Match parameter count where possible, disclose adaptation differences, and report total generated tokens, forward-pass computation estimates, attention volume, and wall-clock time. Equal token counts alone are not equal compute.

### 11.4 Metrics

- Task success and error rates across random seeds.
- Quality versus total compute, latency, and peak accelerator memory.
- Communication volume and useful-information delivery rate.
- Edit acceptance, conflict, rollback, and harmful-edit rates.
- Memory precision, recall, stale retrieval rate, and future-task utility per stored token.
- Resistance to memory poisoning and unauthorized information flow.
- Stream collapse, redundant work, and sensitivity to stream count.

Use paired evaluations and confidence intervals. Inspect intervention experiments, such as removing a stream's contribution, to distinguish useful collaboration from decorative dialogue.

## 12. Minimal viable research prototype

### Scope

- A small transformer trained on bounded synthetic tasks.
- Two fixed agent streams, one shared workspace, and one persistent memory store.
- Shared weights and synchronous decoding rounds.
- Structured attention over visible token blocks.
- Publishing and proposed entry-level edits.
- Task-boundary memory commits and deterministic bounded retrieval.
- Full trace logging and correctness-first cache handling.

The two streams each emit at most one token per round. Workspace and memory windows are initially readable state, not additional independent generators. Later experiments can assign dedicated generator streams to those windows.

### Suggested implementation modules

```text
model/       backbone, stream embeddings, attention masks, decoding
state/       contexts, entries, versions, permissions, commit engine
memory/      durable storage, retrieval, revision, deletion
tasks/       generators, episode definitions, ground-truth checkers
training/    supervised objectives and later outcome optimization
eval/        baselines, ablations, metrics, statistical reports
tracing/     replayable event logs and state inspection
```

This is a proposed layout, not an implementation commitment. Choose framework and hardware targets before scaffolding.

### Milestones and gates

1. **Semantics:** Deterministic replay, visibility isolation, stale-edit rejection, and persistence across restarts all pass tests.
2. **Learning:** A small model learns valid operations and solves distributed-information tasks above no-communication controls.
3. **Collaboration value:** Gains survive comparison against compute-matched baselines and multiple seeds.
4. **Memory value:** Persistence improves held-out future episodes without excessive stale or irrelevant retrieval.
5. **Scaling decision:** Continue only where benefits justify overhead; revise or remove mechanisms that fail ablations.

Numerical acceptance thresholds should be set before each experiment, once task difficulty and resource budgets are established.

## 13. Risks and mitigations

| Risk | Initial mitigation |
| --- | --- |
| All streams converge on identical reasoning. | Distributed private inputs and causal contribution tests; avoid assuming role prompts create specialization. |
| Streams reinforce each other's mistakes. | Track provenance, preserve disagreement, and test independent verification tasks. |
| Communication costs exceed benefits. | Budget communication and compare quality–cost curves. |
| Context edits introduce inconsistent hidden state. | Versioned entries and correctness-first cache recomputation. |
| Memory accumulates errors or sensitive content. | Scoped writes, provenance, revision, retention, and deletion tests. |
| Persistent text becomes an instruction-injection channel. | Fixed permission boundaries and adversarial evaluation. |
| Training exploits synthetic shortcuts. | Diverse generators, held-out distributions, and realistic follow-up tasks. |
| Parallel tokens do not yield practical speedups. | Measure wall-clock performance rather than inferring it from stream count. |
| Credit assignment becomes unstable. | Establish a supervised baseline before adding outcome-based optimization. |

## 14. Open architectural questions

1. Should streams share all weights or have small role-specific adapters?
2. Should shared and memory contexts eventually generate tokens themselves?
3. Is explicit textual communication sufficient, or do latent channels improve efficiency?
4. When are direct peer writes better than proposed edits?
5. Should memory store text, learned state vectors, or both?
6. How should memory retrieval and retention be jointly trained?
7. Does asynchronous scheduling outperform synchronous rounds enough to justify its complexity?
8. Can stream count vary dynamically without collapsing training stability?
9. What is the right unit of memory: token spans, semantic entries, or executable procedures?
10. How should durable memory be validated when no external ground truth is available?

## 15. Related research to investigate

Before making novelty claims, review work on multi-agent learning and communication, shared-workspace architectures, recurrent and memory-augmented transformers, retrieval-augmented language models, persistent agent memory, parallel decoding, and modular computation.

The potential contribution is the combination of **jointly trained multi-stream generation, explicit peer-context mutation, and selective persistent context**. Whether that combination is novel and useful remains to be established through literature review and experiments.

## 16. Initial recommendation

Begin with a synchronous, shared-weight, two-stream transformer with inspectable token-based working contexts and versioned memory. Treat communication, peer editing, and persistence as independently ablatable features.

The first decisive question is not whether the system can produce agent-like conversations. It is whether learned interaction between contexts solves tasks better under a measured resource budget—and whether selected state remains useful after the original computation has ended.
