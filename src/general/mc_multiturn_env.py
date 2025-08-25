from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import random
import re
import copy
import warnings
import torch

from src.base import Environment


@dataclass
class MCExample:
	question: str
	choices: List[str]
	answer_index: int


class MultiTurnMultipleChoiceEnv(Environment):
	"""
	Multi-turn multiple-choice environment over HF datasets (ARC, HellaSwag).

	- No target model; only judge shaping for LL steps and terminal accuracy at submit.
	- State per slot is a list of strings: [formatted_question_and_choices, *history]
	- HL controls when to submit via a special template id: "mc_submit_answer".
	"""

	def __init__(
		self,
		dataset_id: str,
		subset: Optional[str],
		split: str,
		seed: Optional[int],
		max_examples: Optional[int],
		shuffle: bool,
		hf_token: Optional[str],
		batch_size: int,
		max_turns: int = 10,
		judge_model_singleton: Any = None,
		judge_tokenizer: Any = None,
		judge_prompt_config: Optional[Dict[str, Any]] = None,
		# Optional environment-wide generation limits pulled from config
		max_new_tokens: Optional[int] = None,
		warmup_max_new_tokens: Optional[int] = None,
		# Training mode flag - if True, only HL policy is trained (LL agent frozen)
		hl_only_training: bool = False,
		debug_mode: bool = False,
	) -> None:
		try:
			from datasets import load_dataset  # type: ignore
		except Exception as exc:
			raise ImportError("The 'datasets' library is required. Install with: pip install datasets") from exc

		self.dataset_id = dataset_id
		self.subset = subset
		self.split = split
		self.seed = seed
		self.max_examples = max_examples
		self.shuffle = bool(shuffle)
		self.hf_token = hf_token
		self.batch_size = int(batch_size)
		self.max_turns = int(max_turns)

		# Expose generation limits for trainer hooks
		self.max_new_tokens = max_new_tokens if max_new_tokens is not None else 100
		self.warmup_max_new_tokens = warmup_max_new_tokens
		
		# Training mode configuration
		self.hl_only_training = bool(hl_only_training)
		self.debug_mode = bool(debug_mode)

		# Judge model for shaping
		self.judge_model = judge_model_singleton
		self.judge_tokenizer = judge_tokenizer
		self.judge_prompt_config = judge_prompt_config or {}

		# Load dataset and normalize examples
		raw_ds = load_dataset(dataset_id, subset, split=split, token=hf_token)
		examples: List[MCExample] = []

		if "ai2_arc" in dataset_id or dataset_id.endswith("ai2_arc"):
			for ex in raw_ds:
				q = str(ex["question"]).strip()
				choice_items = ex["choices"]["text"]
				choice_labels = ex["choices"]["label"]
				pairs = list(zip(choice_labels, choice_items))
				pairs.sort(key=lambda p: str(p[0]))
				choices = [str(txt).strip() for _, txt in pairs]
				answer_key = str(ex["answerKey"]).strip()
				label_to_index = {str(l): idx for idx, (l, _) in enumerate(pairs)}
				if answer_key not in label_to_index:
					raise ValueError(f"ARC example has unknown answerKey {answer_key!r}")
				answer_index = int(label_to_index[answer_key])
				examples.append(MCExample(q, choices, answer_index))

		elif "hellaswag" in dataset_id.lower():
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
		rng = random.Random(seed)
		if self.shuffle:
			rng.shuffle(examples)
		if max_examples is not None:
			if max_examples <= 0:
				raise ValueError("max_examples must be > 0 if provided")
			examples = examples[:max_examples]

		self._examples: List[MCExample] = examples

		# Per-episode state
		self._indices: List[int] = []
		self._turns: List[int] = []
		self._done: List[bool] = []
		self._histories: List[List[str]] = []

	def reset(self, batch_size: Optional[int] = None) -> List[List[str]]:
		"""Initialize batch slots and return initial states."""
		if batch_size is not None:
			self.batch_size = int(batch_size)
		self._indices = []
		self._turns = [0 for _ in range(self.batch_size)]
		self._done = [False for _ in range(self.batch_size)]
		self._histories = [[] for _ in range(self.batch_size)]

		# Deterministic per-slot sampling based on global seed and slot index
		rng = random.Random(self.seed)
		states: List[List[str]] = []
		for slot in range(self.batch_size):
			idx = rng.randrange(0, len(self._examples))
			self._indices.append(idx)
			formatted = self._format_question(self._examples[idx])
			states.append([formatted])
		# Expose environment state and base_state for trainer compatibility
		self.state = states.copy()
		self.base_state = [s[0] for s in states]
		return states

	def get_base_state(self) -> List[str]:
		"""Return base state strings (formatted question+choices) for each slot."""
		return [self._format_question(self._examples[idx]) for idx in self._indices]

	def get_current_turns(self) -> List[int]:
		"""Return current turn count for each slot."""
		return self._turns.copy()



	def step(
		self,
		actions: List[str],
		**kwargs,
	) -> Tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]:
		"""
		Take a step with LL utterances. HL "submit" is detected via template_ids containing 'mc_submit_answer',
		or via an explicit submit_mask kwarg if provided.
		"""
		update_correctness: bool = kwargs.get("update_correctness", True) # whether or not we finetune the LL model on correctness. Use True for now, ablation later.
		if len(actions) != self.batch_size:
			raise ValueError(f"Batch size mismatch: got {len(actions)} actions, expected {self.batch_size}")

		template_ids: Optional[List[str]] = kwargs.get("template_ids")
		submit_mask: Optional[List[bool]] = kwargs.get("submit_mask")
		system_prompt: Optional[List[str]] = kwargs.get("system_prompt")
		if isinstance(system_prompt, str):
			system_prompt = [system_prompt] * self.batch_size

		# Determine which slots are submitting now
		is_submit: List[bool] = [False] * self.batch_size
		if submit_mask is not None:
			if len(submit_mask) != self.batch_size:
				raise ValueError("submit_mask length must match batch size")
			is_submit = [bool(x) for x in submit_mask]
		elif template_ids is not None:
			if len(template_ids) != self.batch_size:
				raise ValueError("template_ids length must match batch size")
			for i, tid in enumerate(template_ids):
				is_submit[i] = (str(tid or "").strip() == "mc_submit_answer")

		next_states: List[List[str]] = []
		responses: List[str] = []
		hl_rewards: List[float] = []
		ll_rewards: List[float] = []
		dones: List[bool] = []

		for i in range(self.batch_size):
			if self._done[i]:
				# Keep terminal slots unchanged; score zero and done
				next_states.append([self._format_question(self._examples[self._indices[i]]), *self._histories[i]])
				responses.append("")
				hl_rewards.append(0.0)
				ll_rewards.append(0.0)
				dones.append(True)
				continue



			utterance = actions[i].strip()
			if not is_submit[i]:
				# Non-terminal shaping step
				self._histories[i].append(utterance)
				self._turns[i] += 1

				# Judge shaping - use judge scoring for reasoning quality
				ex = self._examples[self._indices[i]]
				instruction = (system_prompt[i] if system_prompt is not None else None)
				score = self._judge_score(ex, self._histories[i], utterance, instruction)
				hl_reward = 0.0

				# Append state and signals
				next_states.append([self._format_question(ex), *self._histories[i]])
				responses.append(utterance)
				hl_rewards.append(hl_reward)
				ll_rewards.append(score)
				dones.append(False)
			else:
				# Submission step - HL reward should only be given when answer is correct
				ex = self._examples[self._indices[i]]
				# Judge shaping - use judge scoring for reasoning quality
				instruction = (system_prompt[i] if system_prompt is not None else None)
				ll_reward = self._judge_score(ex, self._histories[i], utterance, instruction)
				try:
					choice_index = self._parse_choice_index(utterance, num_choices=len(ex.choices))
				except Exception as e:
					#warnings.warn(f"Submit expected a single-letter or 1-4 answer but got: {utterance!r}")
					#print(f"[MCQ parse_error] slot={i}, action={utterance!r}")
					choice_index = -1
				is_correct = int(choice_index == ex.answer_index)
				# HL reward: 0 unless HL submits AND answer is correct
				hl_reward = 1.0 if is_correct else -1.0# This is the proper HL reward
				ll_reward = hl_reward if update_correctness else ll_reward
				if choice_index == -1:
					ll_reward = -1.0
				self._histories[i].append(utterance)
				
				next_states.append([self._format_question_with_choices(ex), *self._histories[i]])
				hl_rewards.append(hl_reward)
				ll_rewards.append(ll_reward)
				dones.append(True)
				self._done[i] = True

		# Expose environment state for trainer compatibility
		self.state = next_states.copy()
		# base_state remains original formatted question strings
		self.base_state = [self._format_question(self._examples[idx]) for idx in self._indices]



		return next_states, responses, torch.tensor(hl_rewards, dtype=torch.float32), torch.tensor(ll_rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.bool)

	def score_safety(self, responses: List[str], **kwargs) -> torch.Tensor:
		"""
		Judge-based shaping score for proposed LL utterances before stepping.
		We ignore 'responses' and instead use kwargs:
		- questions: List[str] -> proposed LL utterances
		- reference_state: List[str] -> formatted question+choices strings.
		- instructions: List[str] -> most recent HL instruction for each sample (optional)

		The judge assesses if the utterance fulfills the most recent HL subgoal.
		"""
		questions: Optional[List[str]] = kwargs.get("questions")
		reference_state: Optional[List[str]] = kwargs.get("reference_state")
		instructions: Optional[List[str]] = kwargs.get("instructions")
		if questions is None or reference_state is None:
			return torch.zeros(len(responses), dtype=torch.float32)
		if len(questions) != len(reference_state):
			raise ValueError("questions and reference_state must be the same length")
		if instructions is not None and len(instructions) != len(questions):
			raise ValueError("instructions must be the same length as questions if provided")

		scores: List[float] = []
		for idx, (ref, utter) in enumerate(zip(reference_state, questions)):
			question, choices = self._split_formatted_question(ref)
			ex = MCExample(question, choices, answer_index=0)  # answer index not needed for judging
			instruction = instructions[idx] if instructions is not None else None
			score = self._judge_score(ex, history=[], new_utterance=str(utter), instruction=instruction)
			scores.append(score)
		return torch.tensor(scores, dtype=torch.float32)

	# ---- Compatibility with trainer interfaces (no target/guard) ----
	def get_target_responses_from_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> List[str]:
		# No target model; return empty strings
		batch = int(input_ids.shape[0]) if hasattr(input_ids, 'shape') else len(input_ids)
		return ["" for _ in range(batch)]

	def get_target_responses(self, actions: List[str], include_conversation_history: bool = False) -> List[str]:
		# No target model; return empty strings
		return ["" for _ in actions]

	def get_target_tokenizer(self):
		# MCQ env has no target tokenizer by design
		return None

	def compute_safety_logits(self, texts: List[str]) -> torch.Tensor:
		# Optional: not used; return None-equivalent by raising to fail-fast if called
		raise AttributeError("compute_safety_logits is not available in MultiTurnMultipleChoiceEnv")

	def _get_embed_similarity_batched(self, sequences1: List[str], sequences2: List[str]) -> torch.Tensor:
		"""
		Compute similarity via the judge model encoder as a proxy since there is no target/guard.
		Returns cosine similarity in [0,1].
		"""
		if self.judge_model is None or self.judge_tokenizer is None:
			raise ValueError("Judge model/tokenizer not provided for MultiTurnMultipleChoiceEnv")
		# Resolve underlying HF model (BaseModelSingleton support)
		base_model = getattr(self.judge_model, 'base_model', self.judge_model)
		# Determine device using robust approach similar to policies
		try:
			device = base_model.get_input_embeddings().weight.device
		except Exception:
			device = next(base_model.parameters()).device
		enc1 = self.judge_tokenizer(sequences1, return_tensors="pt", padding=True, truncation=True, max_length=512)
		enc2 = self.judge_tokenizer(sequences2, return_tensors="pt", padding=True, truncation=True, max_length=512)
		enc1 = {k: v.to(device) for k, v in enc1.items()}
		enc2 = {k: v.to(device) for k, v in enc2.items()}
		with torch.no_grad():
			# Use input embedding average as a lightweight embedding proxy
			# Follow the pattern from policies.py to access input embeddings
			try:
				emb_layer = base_model.get_input_embeddings()
			except Exception as e:
				# Try common nesting (e.g., wrappers) before failing
				inner = getattr(base_model, 'model', None)
				if inner is not None and hasattr(inner, 'get_input_embeddings'):
					emb_layer = inner.get_input_embeddings()
				else:
					raise AttributeError("Judge base model does not expose get_input_embeddings(); ensure a HF PreTrainedModel is provided") from e
			emb1 = emb_layer(enc1["input_ids"])  # [B, T, D]
			emb2 = emb_layer(enc2["input_ids"])  # [B, T, D]
			mask1 = enc1["attention_mask"].unsqueeze(-1)
			mask2 = enc2["attention_mask"].unsqueeze(-1)
			avg1 = (emb1 * mask1).sum(dim=1) / (mask1.sum(dim=1).clamp(min=1))
			avg2 = (emb2 * mask2).sum(dim=1) / (mask2.sum(dim=1).clamp(min=1))
			# Cosine similarity mapped to [0,1]
			num = (avg1 * avg2).sum(dim=-1)
			den = avg1.norm(dim=-1) * avg2.norm(dim=-1) + 1e-8
			cos = (num / den).clamp(min=-1.0, max=1.0)
			return (cos + 1.0) / 2.0

	# Utilities

	def _format_question(self, ex: MCExample) -> str:
		"""Format question without answer choices to avoid constant attention to answers."""
		parts: List[str] = []
		parts.append("Question: " + ex.question)
		return "\n".join(parts) + "\n"

	def _format_question_with_choices(self, ex: MCExample) -> str:
		"""Format question with answer choices for submission or when explicitly needed."""
		parts: List[str] = []
		parts.append("Question: " + ex.question)
		for i, choice in enumerate(ex.choices):
			label = chr(ord('A') + i)
			parts.append(f"{label}) {choice}")
		return "\n".join(parts) + "\n"

	def get_question_choices_text(self) -> List[str]:
		"""Return formatted question + choices strings for current batch indices."""
		return [self._format_question_with_choices(self._examples[idx]) for idx in self._indices]

	def _format_submit_feedback(self, ex: MCExample, chosen_index: int, is_correct: int) -> str:
		correct_label = chr(ord('A') + ex.answer_index)
		chosen_label = chr(ord('A') + max(0, min(chosen_index, len(ex.choices) - 1)))
		status = "Correct." if is_correct else "Incorrect."
		return f"{status} You answered {chosen_label}. The correct answer is {correct_label}.\n"

	def _parse_choice_index(self, text: str, num_choices: int) -> int:
		t = (text or "").strip()
		if not t:
			raise ValueError("Empty answer; expected a choice like 'A'..'D' or '1'..'4'")
		lowered = t.lower()
		# Letter-based
		for idx in range(num_choices):
			letter = chr(ord('a') + idx)
			if f" {letter}" in f" {lowered}" or lowered.strip() == letter or lowered.strip().startswith(letter + ")"):
				return idx
		# Number-based (1-indexed)
		m = re.search(r"\b([1-9])\b", lowered)
		if m:
			val = int(m.group(1)) - 1
			if 0 <= val < num_choices:
				return val
		# Could not parse
		raise ValueError("Could not parse a valid choice from answer. Provide a single letter (A..), or a 1-indexed number.")

	def _split_formatted_question(self, text: str) -> Tuple[str, List[str]]:
		lines = [ln for ln in (text or "").splitlines() if ln.strip()]
		question = ""
		choices: List[str] = []
		for ln in lines:
			if ln.startswith("Question:"):
				question = ln[len("Question:"):].strip()
			elif ") " in ln and ln[0].isalpha() and ln[1] == ')':
				# e.g., "A) text"
				choices.append(ln.split(") ", 1)[1].strip())
		return question, choices

	def _judge_score(self, ex: MCExample, history: List[str], new_utterance: str, instruction: Optional[str]) -> float:
		"""
		Use the judge model to score whether the utterance is appropriate/helpful towards solving the MCQ.
		Return a float in [0,1]. Fail fast if model/tokenizer are not provided.
		"""
		if self.judge_model is None or self.judge_tokenizer is None:
			raise ValueError("Judge model/tokenizer not provided for MultiTurnMultipleChoiceEnv")

		# Build a concise rubric prompt
		rubric_system = self.judge_prompt_config.get(
			"system",
			"You are a strict evaluator for instruction following quality.",
		)
		instruction_text = instruction or ""
		prompt_lines: List[str] = []

		if history:
			prompt_lines.append("History:")
			for h in history:
				prompt_lines.append(f"- {h}")
		prompt_lines.append(f"EXAMPLE 1: {instruction_text}")
		prompt_lines.append(f"EXAMPLE 2: {new_utterance}")
		# Make the format highly constrained to elicit a single-token answer
		prompt_lines.append("Task: Decide if the 'EXAMPLE 2' appropriately follows the instruction in 'EXAMPLE 1'.")
		prompt_lines.append("Rules: Output exactly ONE WORD in UPPERCASE: YES or NO.")
		prompt_lines.append("Decision:")
		judge_prompt = "\n".join(prompt_lines)

		chat_template = [
			{"role": "system", "content": rubric_system},
			{"role": "user", "content": judge_prompt}
		]

		# Tokenize and generate minimal output deterministically
		inputs = self.judge_tokenizer.apply_chat_template(chat_template, return_tensors="pt", padding=True, return_dict=True)
		# Support BaseModelSingleton by resolving underlying model
		base_model = getattr(self.judge_model, 'base_model', self.judge_model)
		try:
			device = base_model.get_input_embeddings().weight.device
		except Exception:
			device = next(base_model.parameters()).device
		inputs = inputs.to(device)
		with torch.no_grad():
			# Use a clean GenerationConfig without sampling params to avoid warnings
			gen_cfg = copy.deepcopy(base_model.generation_config)
			gen_cfg.do_sample = False
			gen_cfg.temperature = None
			gen_cfg.top_p = None
			gen_cfg.top_k = None
			gen_cfg.typical_p = None
			gen_cfg.num_beams = 1
			outputs = base_model.generate(
				input_ids=inputs.input_ids,
				attention_mask=inputs.attention_mask,
				max_new_tokens=2,
				generation_config=gen_cfg,
				pad_token_id=self.judge_tokenizer.eos_token_id,
				eos_token_id=self.judge_tokenizer.eos_token_id,
				use_cache=True,
			)
		# Decode generated suffix only
		input_len = int(inputs.input_ids.shape[1])
		gen_tokens = outputs[0][input_len:]
		text = self.judge_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
		if "yes" in text and "no" not in text:
			return 1.0
		elif "no" in text and "yes" not in text:
			return 0.0
		else:
			warnings.warn(f"Unexpected judge output: {text}")
		# Fallback: classify via next-token logits at the prompt tail (robust to verbose outputs)
		with torch.no_grad():
			lm_out = base_model(
				input_ids=inputs.input_ids,
				attention_mask=inputs.attention_mask,
			)
			last_logits = lm_out.logits[:, -1, :]
			def _single_token_ids(cands: List[str]) -> List[int]:
				ids: List[int] = []
				for s in cands:
					toks = self.judge_tokenizer.encode(s, add_special_tokens=False)
					if len(toks) == 1:
						ids.append(int(toks[0]))
				return ids
			yes_ids = _single_token_ids([" yes", "Yes", "YES", "y", " Y"]) 
			no_ids = _single_token_ids([" no", "No", "NO", "n", " N"]) 
			yes_logit = max((float(last_logits[0, i]) for i in yes_ids), default=float('-inf'))
			no_logit = max((float(last_logits[0, i]) for i in no_ids), default=float('-inf'))
			if yes_logit > no_logit:
				return 1.0
			if no_logit > yes_logit:
				return 0.0
		# Default conservative fallback
		return 0.0 