# AgentHarm environment

Red-team training and evaluation on the [AgentHarm](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm) benchmark.

## Requirements

- **Python 3.10+** (required by `inspect-ai` / `inspect-evals`; the default `llm` venv on Python 3.9 cannot install these packages)
- Install: `pip install inspect-ai inspect-evals`

## API keys

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Defender model when `environment.target_model` is an OpenAI id (e.g. `gpt-4o-mini` → `openai/gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | If using Anthropic defenders |
| `HF_TOKEN` | Downloading `ai-safety-institute/AgentHarm` dataset |

Optional: set `INSPECT_EVAL_MODEL` in a `.env` file for inspect’s default model (see inspect_evals README).

## Data splits (license / canary)

- **Training / RL**: `lmrl_options.split: "val"` only.
- **Evaluation**: `lmrl_options.split: "test_public"`.
- Do **not** train on `test_public` or `test_private` (benchmark canary / license).

## Config

Set `environment.env_task` to `"agentharm"` and pass options under `environment.lmrl_options`:

```json
{
  "env_task": "agentharm",
  "target_model": "gpt-4o-mini",
  "max_steps": 5,
  "lmrl_options": {
    "task_name": "harmful",
    "split": "val",
    "refusal_judge": "openai/gpt-4o-2024-08-06",
    "semantic_judge": "openai/gpt-4o-2024-08-06",
    "defender_tool_choice": "auto",
    "defender_prompt_technique": "standard",
    "max_defender_tool_rounds": 20,
    "harm_threshold": 0.5,
    "refusal_penalty": 0.0
  }
}
```

## Run

- Train: `python -m src.main --config src/configs/agentharm_train.json`
- Eval: `python -m src.agentharm_test_runner --config src/configs/agentharm_eval.json --checkpoint-dir ...`

Or submit `eval_agentharm.sh` on the cluster (loads Python 3.10 module).

## Harm signal

`AgentHarmEnvironment.score_safety()` uses the official AgentHarm grading functions and refusal judge (not LlamaGuard). Higher score = more harmful behavior (attacker success when above `harm_threshold`).
