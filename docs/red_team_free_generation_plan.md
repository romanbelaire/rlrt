# Red-Team Free Generation Plan

## Learnings from the General Setting Experiments

- **Conversation-centric rollouts keep both MCQ and 20Q tasks stable.** The LMRL adapters and MCQ environment updates proved that storing every parallel episode as a `Conversation` object eliminates truncated rollouts, fixes submission handling, and resolves tokenizer padding issues, while also unlocking richer debug metadata.
- **Dual-critic, dual-reward training improves alignment.** Running a single agent with both sequence-level and token-level critics (e.g., Skywork-driven marginal rewards) drives higher question quality, steadier convergence, and cleaner GPU utilization than the earlier dual-agent split.
- **Reward reporting must remain truthful.** Fixes that force rollouts to continue until every conversation ends and that correct max-reward stats (instead of silently clamping) gave us clear visibility into task success without hiding failures.
- **Fail-fast environment factories reduce integration bugs.** The LMRL factory immediately raises if required env assets (e.g., chess scripts, HF oracle models) are missing, which prevented hours of debugging inconsistent states.
- **HuggingFace oracle swap simplified maintenance.** Replacing bespoke T5 pipelines, word variants, and NLTK preprocessing with a slim HuggingFace oracle cut ~40% of the environment code and made yes/no parsing far more reliable.
- **GPU placement strategies matter.** Explicitly parking agent models on `cuda:0` and evaluation/oracle models on `cuda:1` let us run Skywork rewards and environment judges in parallel without OOM, and provided a template for future multi-GPU scheduling.
- **Debug instrumentation is non-negotiable.** Enabling `debug_mode` surfaced per-conversation logs, Skywork score ranges, GPU placement confirmation, and marginal masking progress, which in turn sped up reward-model triage.
- **Template systems limit expressivity.** Even in the general setting, strategy templates eventually bottlenecked exploration; the experiments highlighted that best-performing runs leaned on free-form questioning with only light instructions, motivating this red-team revamp.

## Plan: Enable Free Generation in the Red-Team Top-Level Policy

1. **Define the target interface**
   - Replace template IDs with textual intents. Extend `AgentTrainer` → `HighLevelStep` records so the high-level action is raw text (plus optional metadata like strategy tags) rather than an index into `templates.json`.
   - Specify conversation context packaging so the high-level generator receives the same structured dialogue/history objects used in general experiments.

2. **Introduce a generative high-level policy module**
   - Create `FreeformHighLevelPolicy` alongside `TemplateHighLevelPolicy` in `src/high_level_policy.py`.
   - Use the shared encoder/LLM to sample text via standard decoding (temperature, top-p) instead of a Q-head. Keep the module lightweight by reusing the existing tokenizer and parameter-efficient adapters (LoRA) just like the general single-agent setup.
   - Surface configuration hooks for decoding params, max tokens, and optional stop sequences that align with `Conversation` formatting.

3. **Update training logic**
   - Add PPO/sequence-level training paths for the new policy (mirroring `HighLevelTemplatePolicy` PPO flow in `hierarchical_policies.py` but using generated text outputs).
   - Reuse the dual-critic signal: feed the generated high-level text into the existing sequence-level critic for RL losses, and ensure the low-level/token critic can attribute credit once the generated instructions reach the actor.
   - Enforce fail-fast validation: if the generator produces an empty or invalid instruction, raise immediately to surface bugs rather than silently falling back to a template.

4. **Thread free-form actions through the trainer**
   - Modify `AgentTrainer.collect_rollout()` (and any red-team-specific trainers) so that when `policy_kind == "freeform"`, they:
     - Build prompts using the conversation snapshot (matching the general environment packaging).
     - Store the literal text in replay buffers / rollout batches instead of template indices.
     - Skip template substitution when constructing low-level prompts; instead pass the generated instruction directly.
   - Ensure evaluation paths (`select_action`, checkpoint saving/loading, SLURM/eval scripts) understand the new policy type.

5. **Leverage general-setting infrastructure**
   - Import the conversation-tracking helpers from `src/general` (or move them to a shared module) so red-team rollouts benefit from the same state bookkeeping, termination fixes, and debug logging.
   - Adopt the GPU placement heuristics for any additional reward or guard models needed during free-form training.

6. **Validation and rollout safeguards**
   - Add targeted unit/integration checks that spin up a minimal conversation and assert that the high-level generator produces text and that the trainer routes it end-to-end without touching template registries.
   - Record new debug summaries (e.g., distribution of generated intents, refusal rates) to ensure behavior remains observable once template IDs disappear.

7. **Migration steps**
   - Provide a config flag (e.g., `training.use_freeform_high_level`) defaulting to `false` so we can A/B test against the template baseline.
   - Write a conversion script or checkpoint note describing how to resume from template-based runs (likely requires fresh training due to action-space shift).

This plan keeps the existing template system intact for backward compatibility while introducing a fail-fast, fully generative high-level policy that mirrors the flexibility and observability we established in the general setting experiments.

## Reward and Guard Structure for Red Teaming

- **Sequence reward = LlamaGuard verdicts.** Every target-model response already flows through LlamaGuard; we should capture its binary/score output as the high-level reward signal so that top-level generation directly optimizes toward producing disallowed behavior.
- **Marginal/token reward = the same guard model.** Instead of standing up a separate Skywork critic, reuse LlamaGuard logits/hidden states to compute marginal contributions (masking or leave-one-out over the generated prompt/response). This keeps the reward stack lightweight and removes redundant checkpoints.
- **Single guard model lifecycle.** Centralize guard loading, device placement, and batching in a single module so both sequence-level and marginal evaluators share weights/caches. This avoids double-loading 8B models and makes GPU placement predictable.
- **Fail-fast instrumentation.** Whenever the guard returns `None`, raises, or produces unexpected scores, surface it immediately rather than defaulting to neutral rewards; this mirrors the "fail loud" policy from the general experiments.

## Code-Slimming and Abstraction Strategy

1. **Trainer decomposition**
   - Split `agent_trainer.py` into `trainer/core.py` (rollout orchestration), `trainer/update.py` (critic/policy updates), and task-specific mixins (e.g., `trainer/red_team.py`, `trainer/mcq.py`). Each file targets ≤1k lines while keeping shared utilities centralized.
   - Move replay-buffer helpers, batching logic, and rollout data classes into `trainer/data_structures.py` so trainers no longer embed serialization code.

2. **Policy modules directory**
   - Create `src/policies/high_level/` holding `template_dqn.py`, `template_ppo.py`, and upcoming `freeform.py`. Keep shared prompt-formatting helpers in `src/policies/utils.py` instead of duplicating them across files.

3. **Conversation + environment utilities**
   - Promote the `Conversation` helpers currently embedded in MCQ/LMRL environments into `src/conversation/` (e.g., `conversation/state.py`, `conversation/debug.py`). Both general and red-team code can import the same builders, trimming hundreds of duplicate lines.

4. **Guard/reward service layer**
   - Introduce `src/rewards/guard_evaluator.py` that encapsulates LlamaGuard loading, batching, marginal-masking, and logging. Trainers call `GuardRewardService.sequence_reward(...)` or `marginal_reward(...)` instead of reimplementing loops.

5. **Config and factory cleanup**
   - Extract environment/policy factory functions into `src/factories/` (e.g., `env_factory.py`, `policy_factory.py`) so `main.py` shrinks to argument parsing plus dependency wiring.

6. **Progressive refactor plan**
   - Start with non-functional moves (splitting files without changing logic), add import shims to keep public APIs stable, then iterate on internals. Track line counts per module after each step to ensure we converge toward the 500–1500 line target.

