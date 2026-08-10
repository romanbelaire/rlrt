## Multi-turn ARC/HellaSwag plan

### Goals
- Extend single-step HF multiple-choice tasks (ARC, HellaSwag) to multi-turn dialogue between high-level (HL) and low-level (LL) agents.
- No target model. The environment only manages state, termination, and scoring.
- HL emits subgoals; LL generates utterances toward subgoals; environment appends utterances to state. HL can choose a special “submit” action, after which LL generates the final answer; environment evaluates against ground truth and returns terminal reward for HL.
- LL receives task-completion shaping via LLM-as-a-judge on each utterance.
- We need separate template files for both ARC and HellaSwag. Each has four options.

### Environment design
- New class: `MultiTurnMultipleChoiceEnv` implementing `src.base.Environment`.
  - Constructor args:
    - `dataset_id: str` (e.g., `allenai/ai2_arc`, `Rowan/hellaswag`)
    - `subset: Optional[str]` (ARC-Easy, ARC-Challenge; None for HellaSwag)
    - `split: str`, `seed: Optional[int]`, `max_examples: Optional[int]`, `shuffle: bool`, `hf_token: Optional[str]`
    - `max_turns: int` hard cap on multi-turn exchanges before forced submit
    - `judge_model_singleton`, `judge_tokenizer`: model used to score LL utterances (can be the base actor model)
    - `judge_prompt_config`: few-shot style/rubric
  - State per batch slot:
    - `question_text: str`
    - `choices: List[str]`
    - `history: List[str]` (dialogue turns; LL utterances are appended, HL action choices are tracked as well.)
    - `turn: int`, `done: bool`
    - `ground_truth_index: int`
  - Reset:
    - Sample an example per slot. Return `state = [ [formatted_question_and_choices, *history] ]` per slot.
  - Step(actions, system_prompt=None, template_ids=None, submit_mask=None):
    - Actions are LL utterances (one per active slot) produced after HL gives a subgoal (passed in via `system_prompt`).
    - Detect “submit” intent via `template_ids` containing a known submit template id (preferred), or an explicit `submit_mask` kwarg if provided.
    - If not submitting:
      - Append LL utterance to `history`.
      - Compute LL shaping reward via `judge_model` (see Reward design).
      - Return reward_r_ll for LL, `done=False`. HL reward is implicitly shaped via trainer (see Integration).
    - If submitting:
      - Parse the LL utterance into a choice index (reuse parsing from `HFDatasetMultipleChoiceBatchedEnv`).
      - Compare with ground truth; reward `1.0` if correct else `0.0`.
      - Mark `done=True`. Optionally attach a natural-language feedback string as the last observation.
    - Edge cases: exceed `max_turns` → force submit on next step or auto-submit with current best guess (configurable).

### Reward design
- LL shaping (per non-submit step):
  - LLM-as-a-judge returns a scalar in [0, 1] indicating how much the utterance contributes to solving the subtask, conditioned on high level instruction, question, choices, and prior history.
  - Implementation: a minimal rubric prompt that asks for a YES or NO. Map text to 1/0; fail fast on parse errors.
  - Optional normalization: z-score within batch or simple affine scaling; default: use raw [0, 1].
  - Provide this function FOR THE LOW LEVEL on that step, which is used for the marginal reward function.
  - the HL reward for non terminal steps will be 0.
- HL terminal reward (on submit):
  - 1.0 if correct, else 0.0. intermediate HL reward from env is 0.

### Agent and trainer integration
- HL policy:
  - Add a dedicated “submit answer” template with a stable `id` (e.g., `mc_submit_answer`). Trainer already forwards `template_ids` through `environment.step`; env will detect this id to treat the next LL utterance as the final answer.
  - For MCQ settings, we should have an "examine option" template for each option. It will be the same base template ("Look at option {ID}: {content}. What signals that this might be correct?") but needs to be replicated once per number of options. This assumes the number of options is the same for all questions in the dataset.
- LL policy:
  - Called with `instructions` (HL subgoal) and `base_state` (the formatted question + choices) plus current `state` (which includes earlier utterances). This already matches the `policy.act` signature.
- Trainer loop (existing `collect_rollout` flow is compatible):
  - HL selects a strategy/template id each turn.
  - LL generates an utterance with `instructions` set to HL subgoal.
  - Call `env.step(actions=utterances, system_prompt=instructions, template_ids=[chosen_id])`.
  - Continue while not done. If HL chooses submit template id, the returned reward is terminal.

### Answer parsing and formatting
- Reuse the robust choice parsing from `src/general/hf_mc_env.py` to map LL final utterance to an answer index (letters A.., numbers, etc.).
- Question formatting reuses `_format_question` from the same module to produce the initial state message.

### Judge model prompt (sketch)
- Deterministic, minimal-output rubric. Example:
  - System: “You are a strict evaluator for multiple-choice reasoning quality.”
  - User: Provide question, choices, prior history, and the new utterance. Ask: “Given the instruction, does this response appropriately follow instructions? Provide a YES or NO response.”
- Generation config: `max_new_tokens=8`, `temperature=0.0`, stop on newline; parse float; clip to [0,1].

### Config and templates
- Add configs:
  - `src/configs/general_arc_multiturn.json` and `src/configs/general_hellaswag_multiturn.json` with:
    - `environment.use_mcq_env=false`
    - `environment.env_task: "mc_multiturn"` (new)
    - `environment.lmrl_options` replaced by: `dataset_id`, `subset` (ARC), `split`, `max_turns`, `max_examples`, etc.
    - `training.use_per_token_advantages=false`
  - Templates:
    - `src/configs/templates_general_arc.json` and `src/configs/templates_general_hellaswag.json` to include:
      - Several deliberation subgoal templates (reason, eliminate choices, justify evidence...).
      - One submit template with `id: "mc_submit_answer"` whose instruction explicitly tells LL to output a single letter only.

### Implementation tasks
1) New env implementation: `src/general/mc_multiturn_env.py` [IMPLEMENTED]
   - Load HF dataset; normalize ARC/HellaSwag into `(question, choices, answer_index)`.
   - Manage batch slots, histories, max_turns, done flags.
   - Implement `reset`, `step`, `score_safety` (shaping in `step`).
   - Reuse choice parsing and formatting.
2) Factory wiring: [IMPLEMENTED]
   - Extend `src/main.py.build_environment` to handle `env_task == "mc_multiturn"` and construct `MultiTurnMultipleChoiceEnv`.
   - Reuse target model singleton as judge by default.
3) Configs and templates:
   - Add multi-turn ARC and HellaSwag configs and template files as above.
4) Minimal metrics and logging:
   - Track accuracy (HL terminal), average LL judge score per episode, turns to submit. [TBD]
5) Optional safeguards:
   - Enforce final-answer format in submit step (single letter A..N). [Implicit in parser]
   - Hard stop if LL utterance exceeds max length; truncate. [TBD]

### Open questions for you
- Should we detect submit via `template_ids` (recommended) or add a distinct action flag in the trainer-env call?
    the submit will occur when the submission action is proposed by the HL, ie. template_id == 'mc_submit_answer'
- Preferred max number of turns (`max_turns`) before forcing a submit?
    maximum 10 turns.
- Judge model choice: reuse the actor base model as judge, or provide a separate small judge checkpoint?
    use the actor base model.
- Answer submission: enforce “single letter only” strictly, or accept numbers and map like in the current parser? 
    accept mappings, only do enforcement at the environment level. This will ease our finetuning needs.