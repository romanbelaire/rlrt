from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MCExample:
    question: str
    choices: List[str]
    answer_index: int


class HFDatasetMultipleChoiceBatchedEnv:
    """
    Minimal BatchedTextEnv over HF multiple-choice datasets (ARC, HellaSwag).

    - Exhibits single-turn interaction: present question+choices; agent answers; env returns reward and completes.
    - Adheres to LMRL-Gym BatchedTextEnv interface at runtime via duck typing.
    - Fail fast on any import or data errors per repository policy.
    """

    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        subset: Optional[str] = None,
        seed: Optional[int] = None,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        hf_token: Optional[str] = None,
        prompt_prefix: Optional[str] = None,
    ) -> None:
        try:
            # Imported lazily to fail fast if not available
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise ImportError(
                "The 'datasets' library is required for HF-backed environments. Install with: pip install datasets"
            ) from exc

        try:
            from LLM_RL.environment import Text  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Failed to import LMRL-Gym core. Ensure LMRL-Gym is installed and importable."
            ) from exc

        self.Text = Text
        self.dataset_id = dataset_id
        self.subset = subset
        self.split = split
        self.seed = seed
        self.max_examples = max_examples
        self.shuffle = bool(shuffle)
        self.hf_token = hf_token
        self.prompt_prefix = (prompt_prefix or "").strip()

        # Load and normalize examples
        # Some datasets (e.g., ai2_arc) require a non-empty subset name
        if ("ai2_arc" in dataset_id or dataset_id.endswith("ai2_arc")) and not subset:
            raise ValueError("'allenai/ai2_arc' requires 'subset' to be one of {'ARC-Easy', 'ARC-Challenge'}")
        raw_ds = load_dataset(dataset_id, subset, split=split, token=hf_token)
        examples: List[MCExample] = []

        if "ai2_arc" in dataset_id or dataset_id.endswith("ai2_arc"):
            # Features: question, choices:{text,label}, answerKey: letter
            for ex in raw_ds:
                q = str(ex["question"]).strip()
                choice_items = ex["choices"]["text"]
                choice_labels = ex["choices"]["label"]
                # Ensure consistent A.. ordering by sorting on label if provided
                pairs = list(zip(choice_labels, choice_items))
                pairs.sort(key=lambda p: str(p[0]))
                choices = [str(txt).strip() for _, txt in pairs]
                answer_key = str(ex["answerKey"]).strip()
                # Map letter to index in sorted labels
                label_to_index = {str(l): idx for idx, (l, _) in enumerate(pairs)}
                if answer_key not in label_to_index:
                    raise ValueError(f"ARC example has unknown answerKey {answer_key!r}")
                answer_index = int(label_to_index[answer_key])
                examples.append(MCExample(q, choices, answer_index))

        elif "hellaswag" in dataset_id.lower():
            # Features (Rowan/hellaswag): ctx, endings(list[str]), label(int)
            for ex in raw_ds:
                ctx = str(ex.get("ctx", ex.get("context", ""))).strip()
                endings = [str(e).strip() for e in ex["endings"]]
                label = int(ex["label"])
                question = ctx
                examples.append(MCExample(question, endings, label))

        else:
            raise ValueError(f"Unsupported dataset_id for MC environment: {dataset_id}")

        if len(examples) == 0:
            raise RuntimeError("Loaded zero examples for the requested dataset/split")

        # Optionally cap and shuffle
        import random
        rng = random.Random(seed)
        if self.shuffle:
            rng.shuffle(examples)
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be > 0 if provided")
            examples = examples[:max_examples]

        self._examples: List[MCExample] = examples

        # These are populated on reset based on desired batch size
        self._current_indices: List[int] = []
        self._dones: List[bool] = []

    # Provide copy/close to resemble BatchedTextEnv
    def copy(self) -> "HFDatasetMultipleChoiceBatchedEnv":
        import copy as _copy
        return _copy.deepcopy(self)

    def close(self) -> None:
        return None

    # --- BatchedTextEnv API ---
    def reset(
        self,
        seed: Optional[List[Optional[int]]] = None,
        options: Optional[List[Optional[Dict]]] = None,
    ) -> List[Tuple[Any, ...]]:
        # Length of seeds/options defines batch size
        batch_size = len(seed) if seed is not None else (len(options) if options is not None else 1)

        # Initialize slot pointers deterministically per-slot if seeds provided
        import random
        self._current_indices = []
        self._dones = [False] * batch_size
        histories: List[Tuple[Any, ...]] = []
        for slot in range(batch_size):
            slot_seed = None if seed is None else seed[slot]
            rng = random.Random(slot_seed if slot_seed is not None else self.seed)
            idx = rng.randrange(0, len(self._examples))
            self._current_indices.append(idx)
            histories.append((self.Text(self._format_question(self._examples[idx]), False),))
        return histories

    def step(
        self,
        text_history: List[Optional[Tuple[Any, ...]]],
        done: Optional[List[bool]] = None,
    ) -> List[Optional[Tuple[Tuple[Any, ...], float, bool]]]:
        results: List[Optional[Tuple[Tuple[Any, ...], float, bool]]] = []
        for slot, hist in enumerate(text_history):
            if hist is None:
                results.append(None)
                continue
            already_done = False if done is None else bool(done[slot])
            if already_done or self._dones[slot]:
                results.append(None)
                self._dones[slot] = True
                continue

            # Parse last action from the appended history
            action_text = self._extract_last_action_text(hist)
            if action_text is None:
                raise ValueError("Expected an action appended to the history for each active slot")

            choice_index = self._parse_choice_index(action_text, num_choices=len(self._examples[self._current_indices[slot]].choices))
            ex = self._examples[self._current_indices[slot]]
            is_correct = int(choice_index == ex.answer_index)
            reward = float(is_correct)
            feedback = self._format_feedback(ex, choice_index, is_correct)

            new_hist = hist + (self.Text(feedback, False),)
            self._dones[slot] = True
            results.append((new_hist, reward, True))

        return results

    # --- Helpers ---
    def _format_question(self, ex: MCExample) -> str:
        parts: List[str] = []
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        parts.append("Question: " + ex.question)
        parts.append("Choices:")
        for i, c in enumerate(ex.choices):
            label = chr(ord('A') + i)
            parts.append(f"{label}) {c}")
        # Describe expected answer format based on number of choices
        if len(ex.choices) > 0:
            last_label = chr(ord('A') + len(ex.choices) - 1)
            parts.append(f"Answer with a single letter in [A..{last_label}].")
        else:
            parts.append("Answer with a single letter.")
        return "\n".join(parts) + "\n"

    def _format_feedback(self, ex: MCExample, chosen_index: int, is_correct: int) -> str:
        correct_label = chr(ord('A') + ex.answer_index)
        chosen_label = chr(ord('A') + max(0, min(chosen_index, len(ex.choices) - 1)))
        status = "Correct." if is_correct else "Incorrect."
        return (
            f"{status} You answered {chosen_label}. The correct answer is {correct_label}.\n"
        )

    def _extract_last_action_text(self, history: Tuple[Any, ...]) -> Optional[str]:
        # Walk from end to find last item marked as action
        for item in reversed(history):
            if getattr(item, "is_action", False):
                return str(getattr(item, "text", ""))
        return None

    def _parse_choice_index(self, text: str, num_choices: int) -> int:
        t = (text or "").strip()
        if not t:
            raise ValueError("Empty answer; expected a choice like 'A'..'D' or '1'..'4'")

        # Accept patterns like 'A', 'B.', '(C)', 'option D', '2', 'choice 3', or exact choice text
        lowered = t.lower()

        # 1) Letter-based
        for idx in range(num_choices):
            letter = chr(ord('a') + idx)
            if f" {letter}" in f" {lowered}" or lowered.strip() == letter or lowered.strip().startswith(letter + ")"):
                return idx

        # 2) Number-based (1-indexed)
        import re
        m = re.search(r"\b([1-9])\b", lowered)
        if m:
            val = int(m.group(1)) - 1
            if 0 <= val < num_choices:
                return val

        # 3) Could not parse
        raise ValueError(
            "Could not parse a valid choice from answer. Provide a single letter (A..), or a 1-indexed number."
        )

