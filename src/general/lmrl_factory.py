from __future__ import annotations

from typing import Any, Dict

from src.general.lmrl_adapter import LMRLGymEnvAdapter


def build_lmrl_environment(task: str, batch_size: int, options: Dict[str, Any]) -> LMRLGymEnvAdapter:
    """
    Build an LMRL-Gym environment for a specified task and wrap it with our adapter.

    Args:
        task: One of {"maze", "wordle", "chess", "text_nav"}
        batch_size: Number of parallel instances
        options: Free-form options passed to the underlying task builder.

    Returns:
        LMRLGymEnvAdapter instance.
    """
    task_lc = (task or "").lower()
    if task_lc == "maze":
        from llm_rl_scripts.maze.env.maze_utils import setup_maze_env
        maze_name = options.get("maze_name", "umaze")
        describe_function = options.get("describe_function", "describe_observation_give_position")
        reward_function = options.get("reward_function", "standard_reward")
        max_steps = int(options.get("max_steps", 100))
        last_k = int(options.get("last_k", 40))
        env = setup_maze_env(
            maze_name=maze_name,
            describe_function=describe_function,
            reward_function=reward_function,
            last_k=last_k,
            max_steps=max_steps,
        )
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="maze", task_options=options)

    if task_lc == "wordle":
        from llm_rl_scripts.wordle.env.env import WordleEnvironment
        from llm_rl_scripts.wordle.env.game import Vocabulary
        vocab_path = options.get("vocab_path")
        if not vocab_path:
            raise ValueError("wordle requires 'vocab_path' in options")
        vocab = Vocabulary.from_file(vocab_path)
        require_in_vocab = bool(options.get("require_words_in_vocab", True))
        bad_word_reward = float(options.get("bad_word_reward", -1.0))
        env = WordleEnvironment(vocab=vocab, require_words_in_vocab=require_in_vocab, bad_word_reward=bad_word_reward)
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="wordle", task_options=options)

    # HF datasets as single-turn multiple-choice environments
    if task_lc in {"allenai/ai2_arc", "ai2_arc", "arc"}:
        from src.general.hf_mc_env import HFDatasetMultipleChoiceBatchedEnv
        split = options.get("split", "train")
        subset = options.get("subset", None)  # e.g., "ARC-Easy" or "ARC-Challenge"
        max_examples = options.get("max_examples")
        shuffle = bool(options.get("shuffle", True))
        hf_token = options.get("hf_token")
        prompt_prefix = options.get("prompt_prefix")
        env = HFDatasetMultipleChoiceBatchedEnv(
            dataset_id="allenai/ai2_arc",
            split=split,
            subset=subset,
            seed=options.get("seed"),
            max_examples=max_examples,
            shuffle=shuffle,
            hf_token=hf_token,
            prompt_prefix=prompt_prefix,
        )
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="hf_mc_arc", task_options=options)

    if task_lc in {"rowan/hellaswag", "hellaswag"}:
        from src.general.hf_mc_env import HFDatasetMultipleChoiceBatchedEnv
        split = options.get("split", "train")
        max_examples = options.get("max_examples")
        shuffle = bool(options.get("shuffle", True))
        hf_token = options.get("hf_token")
        prompt_prefix = options.get("prompt_prefix")
        env = HFDatasetMultipleChoiceBatchedEnv(
            dataset_id="Rowan/hellaswag",
            split=split,
            subset=None,
            seed=options.get("seed"),
            max_examples=max_examples,
            shuffle=shuffle,
            hf_token=hf_token,
            prompt_prefix=prompt_prefix,
        )
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="hf_mc_hellaswag", task_options=options)

    if task_lc == "chess":
        # Wrapper expects the chess env in LMRL scripts; if not present, fail-fast
        try:
            from llm_rl_scripts.chess.env.env import FenChessHistoryEnvSingleTurn  # type: ignore
            from LLM_RL.environment import Text
        except Exception as exc:
            raise ImportError("Chess environment not found. Ensure LMRL chess scripts are available and dependencies installed.") from exc
        # Build a minimal initial history (state only) expected by FenChessHistoryEnvSingleTurn
        initial_obs = options.get("initial_observation", "")
        initial_history = (Text(str(initial_obs), False),) if initial_obs is not None else (Text("", False),)
        max_moves = int(options.get("max_moves", 200))
        from_position = options.get("from_position")
        env = FenChessHistoryEnvSingleTurn(initial_history=initial_history, max_moves=max_moves, from_position=from_position)
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="chess", task_options=options)

    if task_lc == "text_nav":
        try:
            from llm_rl_scripts.text_nav.env.env import TextNavEnvironment  # type: ignore
        except Exception as exc:
            raise ImportError("Text-Nav environment not found. Install TextWorld and ensure scripts are available.") from exc
        env = TextNavEnvironment(**{k: v for k, v in options.items() if k not in {"seed", "env_options"}})
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="text_nav", task_options=options)

    if task_lc in {"twenty_questions", "20_questions", "20qs"}:
        # Build twenty questions env with default word list and oracle
        from llm_rl_scripts.twenty_questions.env.data import get_default_word_list
        from llm_rl_scripts.twenty_questions.env.env import TwentyQuestionsPolicyEnvironment
        from llm_rl_scripts.twenty_questions.env.oracle import TwentyQuestionsOracle

        word_list = get_default_word_list()
        max_len = int(options.get("max_conversation_length", 20))

        # Prefer intended T5 oracle if configuration provided
        t5_path = options.get("t5_model_load_path")
        if t5_path:
            try:
                import jax
                from JaxSeq.models.T5.load import ModelLoadMode as T5ModelLoadMode
                from llm_rl_scripts.twenty_questions.env.oracle import T5Oracle
            except Exception as exc:
                raise ImportError("T5 oracle requested but JAX/JaxSEQ are not installed or importable.") from exc

            # Build a simple 1D mesh across all available devices
            devices = jax.devices()
            if len(devices) == 0:
                raise RuntimeError("No JAX devices available for T5 oracle")
            mesh = jax.sharding.Mesh(devices, ("data",))

            prng_key = None
            tokenizer_name = options.get("t5_tokenizer_name", "google/flan-t5-xl")
            load_mode_str = options.get("t5_model_load_mode", "HF").upper()
            try:
                load_mode = getattr(T5ModelLoadMode, load_mode_str)
            except AttributeError as exc:
                raise ValueError(f"Invalid t5_model_load_mode: {load_mode_str}") from exc

            use_fp16_activations = bool(options.get("t5_use_fp16_activations", False))
            use_fp16_params = bool(options.get("t5_use_fp16_params", False))
            max_input_length = int(options.get("t5_max_input_length", 124))
            max_output_length = int(options.get("t5_max_output_length", 4))

            oracle = T5Oracle.load_oracle(
                mesh=mesh,
                prng_key=prng_key,
                model_load_mode=load_mode,
                model_load_path=t5_path,
                tokenizer_name=tokenizer_name,
                use_fp16_activations=use_fp16_activations,
                use_fp16_params=use_fp16_params,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
            )
        else:
            # Minimal stub oracle implementation using simple heuristics
            class SimpleYesNoOracle(TwentyQuestionsOracle):
                def generate_answers(self, words, questions, return_full: bool=False):
                    # Always answer "No." unless the question directly equals the word with a leading 'Is it'
                    def answer_for(word, q):
                        q = (q or "").strip().lower().rstrip("?")
                        if isinstance(word, list):
                            target = str(word[0]).lower()
                        else:
                            target = str(word).lower()
                        # Very naive containment heuristic
                        return "Yes." if target in q else "No."
                    if isinstance(words, list) and isinstance(questions, list):
                        return [answer_for(w, q) for w, q in zip(words, questions)]
                    return answer_for(words, questions)
            oracle = SimpleYesNoOracle()

        env = TwentyQuestionsPolicyEnvironment(oracle=oracle, word_list=word_list, max_conversation_length=max_len)
        return LMRLGymEnvAdapter(env, batch_size=batch_size, task="twenty_questions", task_options=options)

    raise ValueError(f"Unknown LMRL task: {task}") 