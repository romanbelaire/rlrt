# WildJailbreak Data Loading for Conversation Format

This document describes how WildJailbreak data is loaded and used for conversation format, for use in training the red teamer (attacker/PatientAgent) to elicit target prompts from the defender agent.

---

## 1. Dataset Source

- **HuggingFace**: `allenai/wildjailbreak`
- **Splits**: `train` or `eval` (config: `wildjailbreak_split`)
- **Load call**: `load_dataset("allenai/wildjailbreak", split, delimiter="\t", keep_default_na=False)[split]`

---

## 2. Raw Row Fields

Each row in the dataset has:

| Field        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `data_type`  | One of: `vanilla_harmful`, `vanilla_benign`, `adversarial_harmful`, `adversarial_benign` |
| `vanilla`    | Direct prompt string (the base question/request)                             |
| `adversarial`| Jailbreak-style prompt string; may be empty                                 |
| `completion` | Optional; not used in conversation flow                                   |

---

## 3. Prompt Selection (Target Goal)

The **attacker goal** is the prompt the red teamer must eventually elicit:

```
prompt = adversarial if adversarial else vanilla
```

- When `adversarial` is non-empty: use it (jailbreak variant).
- When `adversarial` is empty: use `vanilla` (direct prompt).

`harmful_level` is derived from `data_type`:

```
harmful_level = 1 if "harmful" in data_type else 0
```

---

## 4. Example Dict Produced by Loader

Each loaded example has:

| Key            | Value                                                                 |
|----------------|----------------------------------------------------------------------|
| `id`           | `wildjailbreak_{split}_{idx}`                                        |
| `base_prompt`  | The target prompt (adversarial or vanilla)                           |
| `prompt`       | Same as `base_prompt`                                                |
| `harmful_level`| 0 (benign) or 1 (harmful)                                            |
| `data_type`    | Original `data_type` string                                          |
| `completion`   | From raw row (optional)                                               |
| `lead_in`      | `"I have a question."` (used in offline mode)                         |
| `goal_json`    | `{"base_prompt": prompt}` (for PatientAgent / UserSimulator)          |

---
