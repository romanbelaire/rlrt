# CARES-18K Data Loading for Conversation Format

This document describes how CARES-18K data is loaded and used for conversation format, for use in training the patient simulator (PatientAgent) to elicit target prompts from the healthcare assistant agent.

---

## 1. Dataset Source

- **HuggingFace**: `HFXM/CARES-18K`
- **Splits**: `train` or `test` (config: `cares_split`)
- **Load call**: `load_dataset("HFXM/CARES-18K", split=split)`

---

## 2. Raw Row Fields

Each row in the dataset has:

| Field             | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `base_prompt`     | Original direct prompt before manipulation (the semantic goal)              |
| `prompt`          | Final version of the prompt (possibly adversarially rewritten)              |
| `method`          | Prompting method: `direct`, `indirect`, `obfuscate`, or `role_play`         |
| `harmful_level`   | Harmfulness level: 0 (harmless) to 3 (highly harmful)                       |
| `generation_model`| LLM used to generate the prompt (e.g., GPT, Claude, Gemini, Deepseek)       |
| `principle_index` | Index of the safety principle (0–7)                                         |

---

## 3. Prompt Selection (Target Goal)

CARES uses two distinct prompt fields:

- **`base_prompt`**: The semantic goal the patient/attacker must eventually elicit. Used for `goal_json` and PatientAgent.
- **`prompt`**: The actual text shown to the agent. May be rewritten via `method` (direct, indirect, obfuscation, role-play).

For **online mode** (PatientAgent): the goal is `base_prompt`; the PatientAgent generates messages toward eliciting that question.

For **offline mode**: fixed turns use `prompt` as the second user message (the one the agent responds to for reward).

---

## 4. Example Dict Produced by Loader

Each loaded example has:

| Key               | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| `id`              | `cares_{split}_{idx}`                                                 |
| `base_prompt`     | Original direct prompt (semantic goal)                                |
| `prompt`          | Final prompt text (possibly rewritten)                               |
| `harmful_level`   | 0–3 (0 = harmless, 3 = highly harmful)                                 |
| `principle_index` | 0–7 (safety principle index)                                         |
| `generation_model`| From raw row                                                          |
| `method`          | `direct`, `indirect`, `obfuscate`, or `role_play`                     |
| `goal_json`       | `{"base_prompt": base_prompt}` (for PatientAgent / UserSimulator)     |

**Note**: CARES does not set `lead_in` in the loader. The environment defaults to `"Hi, I have a health-related question."` when `lead_in` is missing.

---

## 5. Goal Format for Patient Simulator

The PatientAgent receives `goal_json`:

- **Format**: `{"base_prompt": "<base_prompt>"}`
- **Natural language**: `format_multiwoz_goal(goal_json)` returns the `base_prompt` string.
- **Usage**: Injected into patient templates as `{base_prompt}` and `{goal_text}` (both resolve to the same string).

---

## 6. Conversation Flow

### 6.1 Online Mode (default)

- **PatientAgent** generates multi-turn patient messages.
- **Goal**: Progressively elicit the target `base_prompt` from the healthcare assistant.
- **Template**: Uses `patient_with_goal` (or `patient_strategist`) from `cares_patient_prompts.json`; no harmful/benign template split (unlike WildJailbreak).
- **Dialogue history**: List of `(agent_action, user_response)` pairs.
- **Initial turn**: PatientAgent generates first user message from empty history and empty `agent_action`.

### 6.2 Offline Mode

- Fixed turns: `[lead_in, prompt]` or `[prompt]` (depending on `cares_use_lead_in`).
- **Lead-in**: `"Hi, I have a health-related question."` (default when not set).
- Turn 1: User says `lead_in` → agent responds.
- Turn 2: User says `prompt` (the rewritten prompt, not base_prompt) → agent responds → reward computed.

---

## 7. Patient Prompt Templates

Templates live in `src/prompts/cares_patient_prompts.json`:

- **`patient_with_goal`**: Default; healthcare context, gradual revelation toward goal question.
- **`patient_strategist`**: Shorter variant; same gradual-revelation constraint.

Placeholders:

- `{base_prompt}` / `{goal_text}`: Target question to eventually ask.
- `{state_text}`: Formatted dialogue history (`Assistant: ...`, `Patient: ...`).
- `{agent_action}`: Assistant's last message (or `"(Conversation start)"` on turn 0).

**Constraint**: Do not state the full goal in the first message(s). Build toward it gradually over several turns. Steer back if the assistant goes off-topic (e.g., restaurants, travel).

---

## 8. Dialogue History Format

For the patient simulator, dialogue history is a list of `(agent_action, user_response)` tuples:

- `agent_action`: Canonical assistant utterance (no system instructions).
- `user_response`: Patient message.

Formatted for templates:

```
Assistant: <agent_action>
Patient: <user_response>
```

---

## 9. Config Options

| Config Key          | Default   | Description                                                       |
|---------------------|-----------|-------------------------------------------------------------------|
| `cares_split`       | `"train"` | Dataset split (`train` or `test`)                                 |
| `cares_use_lead_in` | `True`   | Use lead-in in offline mode                                        |
| `cares_online`      | `True`   | Use PatientAgent (online) vs fixed turns (offline)                |
| `max_dialogues`     | -         | Cap number of examples (random sample if set)                     |

---

## 10. Training Implications for Patient Simulator

1. **Goal representation**: Patient simulator must treat `base_prompt` as the target to elicit; it is passed via `goal_json`.
2. **Gradual revelation**: Templates instruct the patient to build context over turns and not ask the full question immediately.
3. **base_prompt vs prompt**: `base_prompt` is the semantic goal; `prompt` is the actual text (possibly rewritten). In online mode, the patient works toward `base_prompt`; in offline mode, the fixed second turn uses `prompt`.
4. **Harmful levels 0–3**: Unlike WildJailbreak (0/1), CARES uses 0–3. Reward and termination logic use this for LlamaGuard (harmful) vs fulfillment judge (benign).
5. **Method field**: `direct`, `indirect`, `obfuscate`, `role_play` describe how the prompt was generated; the patient simulator does not receive this, but it may inform future training (e.g., teaching the patient to use different strategies).
