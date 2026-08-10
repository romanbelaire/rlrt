"""
Factory functions for building hierarchical 20 Questions environments.

This module provides factory functions to create hierarchical 20Q environments
with proper oracle and judge model integration.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hierarchical_twentyq_env import HierarchicalTwentyQuestionsEnvironment, TwentyQuestionsOracle
from src.general.lmrl_adapter import LMRLGymEnvAdapter


class HuggingFaceOracle(TwentyQuestionsOracle):
    """
    HuggingFace-based oracle for hierarchical 20 Questions.
    
    This oracle uses a HuggingFace language model to answer yes/no questions
    about target words.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantize: Optional[Union[bool, int]] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the HuggingFace oracle.
        
        Args:
            model_name: HuggingFace model name
            quantize: Quantization settings (4, 8, or None)
            device: Device to place model on
            hf_token: HuggingFace authentication token
            debug_mode: Enable debug logging
        """
        self.model_name = model_name
        self.quantize = quantize
        # Use GPU 1 if available, otherwise fall back to GPU 0 or CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = "cuda:1"  # Use GPU 1 when available
            else:
                self.device = "cuda:0"  # Fall back to GPU 0 if only one GPU
        else:
            self.device = "cpu"
        self.debug_mode = debug_mode
        
        # Load model and tokenizer
        self._load_model_and_tokenizer(hf_token)
    
    def _load_model_and_tokenizer(self, hf_token: Optional[str]) -> None:
        """Load the HuggingFace model and tokenizer."""
        if self.debug_mode:
            print(f"[ORACLE] Loading HuggingFace model: {self.model_name}")
        
        # Configure quantization
        quantization_config = None
        if self.quantize:
            from transformers import BitsAndBytesConfig
            if self.quantize == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif self.quantize == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token,
            padding_side='left'  # Left padding for decoder-only models
        )
        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_token,
            quantization_config=quantization_config,
            device_map="auto" if self.quantize else None,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        )
        
        if not self.quantize:
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def generate_answers(
        self, 
        words: Union[str, List[str]], 
        questions: Union[str, List[str]], 
        return_full: bool = False
    ) -> Union[str, List[str]]:
        """
        Generate oracle answers for given words and questions.
        
        Args:
            words: Target word(s)
            questions: Question(s) to answer
            return_full: Return full response or just Yes/No
            
        Returns:
            Oracle answer(s)
        """
        # Remember if inputs were originally single
        single_word = isinstance(words, str)
        single_question = isinstance(questions, str)
        
        # Handle single inputs
        if single_word:
            words = [words]
        if single_question:
            questions = [questions]
        
        # Use batched processing for better performance
        answers = self._generate_batched_answers(words, questions, return_full)
        
        # Return single answer if single input
        if len(answers) == 1 and single_word and single_question:
            return answers[0]
        
        return answers
    
    def _generate_batched_answers(self, words: List[str], questions: List[str], return_full: bool) -> List[str]:
        """Generate oracle answers using batched processing for better performance."""
        if len(words) == 0:
            return []
        
        # Build all prompts
        prompts = []
        for word, question in zip(words, questions):
            prompt = self._build_oracle_prompt(word, question)
            prompts.append(prompt)
            
            if self.debug_mode:
                print(f"[ORACLE] Word: '{word}', Question: '{question}'")
                print(f"[ORACLE] Prompt: {prompt[:100]}...")
        
        try:
            # Batch tokenization
            if self.debug_mode:
                print(f"[ORACLE] Processing {len(prompts)} questions in a single batch (was: {len(prompts)} individual calls)")
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                padding_side='left'
            ).to(self.device)
            
            # Single batched generation call
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3,  # Short response
                    do_sample=False,   # Deterministic
                    num_beams=1,       # No beam search
                    temperature=None,  # Explicitly set to None to override model's default generation config
                    top_p=None,        # Explicitly set to None to override model's default generation config
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Process each response in the batch
            answers = []
            for i in range(len(words)):
                # Decode response for this sample
                response = self.tokenizer.decode(
                    outputs[i][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                if self.debug_mode:
                    print(f"[ORACLE] Raw response: '{response}'")
                
                # Parse response
                parsed_answer = self._parse_oracle_response(response)
                
                if self.debug_mode:
                    print(f"[ORACLE] Parsed answer: '{parsed_answer}'")
                
                final_answer = response if return_full else parsed_answer
                answers.append(final_answer)
            
            return answers
        
        except Exception as e:
            if self.debug_mode:
                print(f"[ORACLE] Error generating batched answers: {e}")
            # Fallback to individual processing if batching fails
            return [self._generate_single_answer(word, question, return_full) 
                    for word, question in zip(words, questions)]
    
    def _generate_single_answer(self, word: str, question: str, return_full: bool) -> str:
        """Generate a single oracle answer."""
        # Build prompt
        prompt = self._build_oracle_prompt(word, question)
        
        if self.debug_mode:
            print(f"[ORACLE] Word: '{word}', Question: '{question}'")
            print(f"[ORACLE] Prompt: {prompt[:100]}...")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                padding_side='left'
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3,  # Short response
                    do_sample=False,   # Deterministic
                    num_beams=1,       # No beam search
                    temperature=None,  # Explicitly set to None to override model's default generation config
                    top_p=None,        # Explicitly set to None to override model's default generation config
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            if self.debug_mode:
                print(f"[ORACLE] Raw response: '{response}'")
            
            # Parse response
            parsed_answer = self._parse_oracle_response(response)
            
            if self.debug_mode:
                print(f"[ORACLE] Parsed answer: '{parsed_answer}'")
            
            return response if return_full else parsed_answer
        
        except Exception as e:
            if self.debug_mode:
                print(f"[ORACLE] Error generating answer: {e}")
            return "No."  # Conservative default
    
    def _build_oracle_prompt(self, word: str, question: str) -> str:
        """Build the oracle prompt for a word-question pair."""
        return f"""You are the oracle for a 20 questions game. Your word is "{word}".

Answer the following question about the word with ONLY a YES or a NO:

Question: {question}

IMPORTANT: Respond with exactly one word - either "YES" or "NO". Do not provide any explanation or additional text.

Answer:"""
    
    def _parse_oracle_response(self, response: str) -> str:
        """Parse the oracle response to extract Yes/No."""
        response_lower = response.lower().strip()
        
        # Direct matches
        if response_lower in ["yes", "y", "true"]:
            return "Yes."
        elif response_lower in ["no", "n", "false"]:
            return "No."
        
        # Check first word
        words = response_lower.split()
        if words:
            first_word = words[0].strip('.,!?')
            if first_word in ["yes", "y", "yep", "yeah", "correct", "true"]:
                return "Yes."
            elif first_word in ["no", "n", "nope", "incorrect", "false"]:
                return "No."
        
        # Conservative default
        return "No."


def build_hierarchical_twentyq_environment(
    batch_size: int = 4,
    options: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False,
    # Oracle configuration
    oracle_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    oracle_quantize: Optional[Union[bool, int]] = None,
    hf_token: Optional[str] = None,
    # Judge model configuration (optional)
    judge_model_name: Optional[str] = None,
    judge_quantize: Optional[Union[bool, int]] = None,
    judge_batch_size: int = 32,
    # Basic environment configuration
    max_conversation_length: int = 20,
    # Reward configuration
    hl_reward_success: float = 0.0,
    hl_reward_failure: float = -1.0,
    ll_reward_range: Tuple[float, float] = (0.0, 1.0),
) -> HierarchicalTwentyQuestionsEnvironment:
    """
    Build a hierarchical 20 Questions environment.
    
    Args:
        batch_size: Number of parallel conversations
        options: Additional options (from config)
        debug_mode: Enable debug logging
        oracle_model_name: HuggingFace model for oracle
        oracle_quantize: Oracle model quantization
        hf_token: HuggingFace token
        judge_model_name: Judge model for instruction-following evaluation (optional)
        judge_quantize: Judge model quantization
        judge_batch_size: Batch size for judge model evaluation
        max_conversation_length: Maximum questions per game
        hl_reward_success: High-level reward for success
        hl_reward_failure: High-level reward for failure
        ll_reward_range: Low-level reward range
        
    Returns:
        Configured hierarchical 20Q environment
    """
    # Parse options
    if options:
        oracle_model_name = options.get("oracle_model_name", oracle_model_name)
        oracle_quantize = options.get("oracle_quantize", oracle_quantize)
        hf_token = options.get("hf_token", hf_token)
        judge_model_name = options.get("judge_model_name", judge_model_name)
        judge_quantize = options.get("judge_quantize", judge_quantize)
        judge_batch_size = options.get("judge_batch_size", judge_batch_size)
        max_conversation_length = options.get("max_conversation_length", max_conversation_length)
        hl_reward_success = options.get("hl_reward_success", hl_reward_success)
        hl_reward_failure = options.get("hl_reward_failure", hl_reward_failure)
        ll_reward_range = options.get("ll_reward_range", ll_reward_range)
    
    # Build oracle
    oracle = HuggingFaceOracle(
        model_name=oracle_model_name,
        quantize=oracle_quantize,
        hf_token=hf_token,
        debug_mode=debug_mode
    )
    
    # Build judge model if specified
    judge_model = None
    judge_tokenizer = None
    
    if judge_model_name:
        if debug_mode:
            print(f"[JUDGE] Loading judge model: {judge_model_name}")
        
        # Configure judge quantization
        judge_quantization_config = None
        if judge_quantize:
            from transformers import BitsAndBytesConfig
            if judge_quantize == 4:
                judge_quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif judge_quantize == 8:
                judge_quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
        
        # Load judge tokenizer
        judge_tokenizer = AutoTokenizer.from_pretrained(
            judge_model_name,
            token=hf_token,
            padding_side='left'
        )
        judge_tokenizer.padding_side = 'left'
        
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        
        # Load judge model
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            token=hf_token,
            quantization_config=judge_quantization_config,
            device_map="auto" if judge_quantize else None,
            torch_dtype=torch.float16
        )
        
        judge_model.eval()
    
    # Define word list
    word_list = [
        # Sports equipment
        ["Basketball", "Basket ball", "B-ball"],
        ["Football", "American football", "Pigskin"],
        ["Baseball", "Base ball"],
        ["Soccer ball", "Football", "Soccer"],
        
        # Animals
        ["Cat", "Kitty", "Feline", "House cat"],
        ["Dog", "Canine", "Puppy", "Hound"],
        ["Horse", "Equine", "Steed"],
        ["Cow", "Bovine", "Cattle"],
        ["Elephant", "Pachyderm"],
        ["Lion", "Big cat", "King of jungle"],
        
        # Fruits
        ["Apple", "Red apple", "Green apple"],
        ["Banana", "Yellow banana"],
        ["Orange", "Citrus fruit"],
        ["Strawberry", "Strawberries"],
        ["Grape", "Grapes"],
        
        # Vehicles
        ["Car", "Automobile", "Vehicle"],
        ["Truck", "Pickup truck", "Semi truck"],
        ["Motorcycle", "Motorbike", "Bike"],
        ["Boat", "Ship", "Vessel"],
        ["Airplane", "Aircraft", "Plane"],
        
        # Household items
        ["Chair", "Seat"],
        ["Table", "Desk"],
        ["Lamp", "Light"],
        ["Book", "Novel"],
        ["Phone", "Telephone", "Mobile phone"],
        
        # Tools
        ["Hammer", "Tool"],
        ["Screwdriver", "Driver"],
        ["Knife", "Blade"],
        ["Scissors", "Shears"],
        
        # Food items
        ["Pizza", "Pie"],
        ["Sandwich", "Sub"],
        ["Cookie", "Biscuit"],
        ["Ice cream", "Frozen dessert"]
    ]
    
    # Create environment
    environment = HierarchicalTwentyQuestionsEnvironment(
        oracle=oracle,
        word_list=word_list,
        max_conversation_length=max_conversation_length,
        debug_mode=debug_mode,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        judge_batch_size=judge_batch_size,
        hl_reward_success=hl_reward_success,
        hl_reward_failure=hl_reward_failure,
        ll_reward_range=ll_reward_range
    )
    
    if debug_mode:
        print(f"[FACTORY] Created hierarchical 20Q environment with {len(word_list)} word variants")
    
    return environment


def build_hierarchical_twentyq_adapter(
    batch_size: int = 4,
    options: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False,
    **kwargs
) -> LMRLGymEnvAdapter:
    """
    Build a hierarchical 20Q environment wrapped in LMRLGymEnvAdapter.
    
    This provides compatibility with existing trainer interfaces.
    
    Args:
        batch_size: Number of parallel conversations
        options: Additional options
        debug_mode: Enable debug logging
        **kwargs: Additional arguments
        
    Returns:
        Wrapped hierarchical 20Q environment
    """
    # Build the core environment
    env = build_hierarchical_twentyq_environment(
        batch_size=batch_size,
        options=options,
        debug_mode=debug_mode,
        **kwargs
    )
    
    # Wrap in adapter for compatibility
    adapter = LMRLGymEnvAdapter(env, batch_size=batch_size)
    
    # Set task identifier
    adapter.task = "hierarchical_twenty_questions"
    
    return adapter
