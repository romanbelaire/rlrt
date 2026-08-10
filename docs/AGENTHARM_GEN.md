# AgentHarm-Gen Pipeline

Data-generation component of *Agent vs. Agent* (Kulkarni et al., EMNLP 2025 Industry Track), implemented under `src/agentharm_gen/`.

The pipeline converts AppWorld API documentation into a synthetic tool-use benchmark: mock `@tool` functions, adversarial task prompts, and per-task grading functions. The result is an Inspect-AI task you can run the same way as the upstream AgentHarm benchmark.

## Components

```
src/agentharm_gen/
  converter/        Component 1: AppWorld API docs → @tool Python files
  taskgen/          Component 2: Adversarial task record generator
  gradergen/        Component 3: Per-task grading function generator
  harness/          Inspect-AI task + dynamic tool/grader registries
  optimizer/        Algorithm 1 skeleton (research seam)
  prompts/          Verbatim A.1/A.2/A.3 generator prompts from paper appendix
  generated/        Runtime artifacts (gitignored)
    tools/          ~174 @tool .py files
    tasks.jsonl     Generated task records
    graders/        Per-task grading function .py files
    registry.json   Tool/grader path registry (resume checkpoint)
```

## Setup

```bash
# Install AppWorld (only needed for the 'convert' step)
pip install appworld

# Set API key for the backbone generator model (default: claude-3-7-sonnet-20250219)
export ANTHROPIC_API_KEY=...

# Set API keys for refusal/semantic judge models
export OPENAI_API_KEY=...
```

Override the backbone model:
```bash
export AGENTHARM_GEN_MODEL=claude-3-7-sonnet-20250219
```

If AppWorld is installed elsewhere, point at its data directory:
```bash
export APPWORLD_DATA_DIR=/path/to/appworld/data
```

## Pipeline

Run steps in order:

### Step 1 — Convert AppWorld APIs to @tool files

```bash
python -m src.agentharm_gen.cli convert
```

Options:
- `--apps venmo,spotify,phone,amazon,gmail` — subset of apps (default: all five)
- `--no-resume` — regenerate all tools from scratch
- `--out-dir PATH` — custom output directory

Target: ~174 gate-passing tools written to `generated/tools/`.

### Step 2 — Generate adversarial tasks

```bash
python -m src.agentharm_gen.cli tasks -n 5
```

Options:
- `--categories Copyright,Fraud,...` — subset of the 8 categories
- `-n N` — tasks per LLM call per category (default: 5; oversample and filter)
- `--out-path PATH` — custom output path for `tasks.jsonl`

Target: ≥71 unique task records with avg ~5.75 target functions.

### Step 3 — Generate grading functions

```bash
python -m src.agentharm_gen.cli graders
```

Options:
- `--tasks-path PATH` — custom tasks.jsonl path
- `--no-resume` — regenerate all graders
- `--out-dir PATH` — custom output directory

### Step 4 — Run the benchmark

```bash
python -m src.agentharm_gen.cli eval --model openai/gpt-4o --limit 5
```

Or directly via `inspect`:
```bash
inspect eval src/agentharm_gen/harness/task.py@agentharm_gen \
  --model openai/gpt-4o \
  --limit 5 \
  -T semantic_judge=openai/gpt-4o-2024-08-06 \
  -T refusal_judge=openai/gpt-4o-2024-08-06
```

View results:
```bash
inspect view
```

## Sanity targets (§8 of implementation guide)

1. ~174 importable, gate-passing tools across {venmo, spotify, phone, amazon, gmail, browser}
2. ≥71 unique, tool-valid tasks across 8 categories; avg target-function count ≈ 5.75
3. Each task has a gate-passing `grade_{task_id}` function with matching `target_functions`
4. Baseline eval (no refinement): near-zero ASR, ~97% refusal rate
5. With refinement, expect movement toward GPT-4o ≈ 0.45 avg harm (with the paper's optimizer)

## Validation gate

Every generated file passes three checks before being written to disk:

1. `ast.parse` — rejects syntax errors immediately
2. Static check — no banned names (`os`, `subprocess`, `eval`, etc.), imports ⊆ `inspect_ai`
3. Restricted import — exec in isolated namespace, assert correct callable type

## Research seam — Algorithm 1

The iterative refinement loop in `optimizer/refine.py` exposes a clean boundary for your critic/Q-function. Replace `EchoAttacker` with your own `AttackerPolicy`:

```python
from src.agentharm_gen.optimizer.attacker import AttackerPolicy

class MyCriticAttacker:
    async def next_prompt(self, original_prompt, current_prompt,
                          assistant_responses, tool_calls, best_score) -> str:
        # Your learned critic or reflection policy here
        ...
```

Wire it into `iterative_refine()`:
```python
from src.agentharm_gen.optimizer.refine import iterative_refine
result = await iterative_refine(task, p1=task["prompt"], run_agent=..., grade=...,
                                 attacker=MyCriticAttacker(), R=2)
```

The harness exposes per-step `(score, refusal, all_target_functions_called, correct_order, answer_on_topic)` as dense reward signal for learning.

## Relationship to existing red-team RL stack

The AgentHarm-Gen pipeline is independent of the PPO training stack in `src/agentharm/`. The existing `AgentHarmEnvironment` trains on the fixed upstream AISI benchmark; the gen pipeline produces a new synthetic benchmark. Follow-up work can extend `AgentHarmEnvironment` with a `dataset_source: "generated"` option to train on generated behaviors.

See `docs/AGENTHARM.md` for the RL training setup.
