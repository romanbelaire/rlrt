from agent import Agent, test_inputs
import argparse
import torch 
from datetime import datetime
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_finetuned_model(base_model_name, adapter_path):
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        base_model_name: The name of the base model used for fine-tuning
        adapter_path: Path to the saved LoRA adapter weights
        
    Returns:
        model: The loaded model with LoRA adapters
        tokenizer: The tokenizer for the model
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the LoRA configuration
    print(f"Loading LoRA adapters from: {adapter_path}")
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Optionally merge the adapters with the base model for faster inference
    if args.merge_adapters:
        print("Merging LoRA adapters with base model...")
        model = model.merge_and_unload()
    
    return model, tokenizer

def main(args):
    hyperparameters = {
        'lr': 5e-5,
        'gamma': 0.99,
        'epsilon': 0.2,
        'batch_size': 64,
        'minibatch_size': 8,
        'epochs': 5,
        'value_lr': 5e-3,
        'episodes': args.episodes,
        'max_timesteps': 3000000,
        'clip': 0.2,
        'model_name': args.model_name,
        'test_model': args.test_model,
        'checkpoint': args.checkpoint,
        'online_eps': args.online_eps,
        'val_type': args.val_type,
        'max_ep_len': 5,
        'prob_granularity': args.prob_granularity,
        'alg': args.alg,
        'max_rollout_len': 64,
        'adv_coef': 0.1,
        'sem_coef': 0.5,
        'help_weight': args.help_weight,
        'override': args.override,
        'attack_type': args.attack_type,
        'detune': args.detune
    }
    if args.test:
        assert args.test_model is not None, "Please specify a model folder to test via --test-model"
        hyperparameters['model_name'] = args.test_model
    else:
        if args.test_model:
            print(f"test_model contains value {args.test_model}, did you mean to enable --test?")
        hyperparameters['test_model'] = None

    hyperparameters.update({
        'train_steps': args.train_steps,
        'data_dir': args.data_dir,
        'train': args.train,
    })

    policy_type = 'Transformer'

    # Create Agent with optional fine-tuned model
    agent = Agent(
        policy_type=policy_type, 
        data_path=args.data_dir, 
        hf_checkpoint=args.checkpoint, 
        hyperparameters=hyperparameters, 
        auth_token="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",
        use_finetuned=args.use_finetuned,
        adapter_path=args.adapter_path,
        merge_adapters=args.merge_adapters
    )
    
    if args.evaluate:
        print(f"Evaluating red-teaming effectiveness using {args.eval_dataset}")
        metrics = agent.evaluate_red_teaming(
            dataset_name=args.eval_dataset,
            num_samples=args.eval_samples
        )
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    elif args.test:
        agent.test(hyperparameters['episodes'])
    else:
        print("Training agent with algorithm: ", args.alg)
        if args.detune:
            print("Detuning model safety alignment...")
            agent.detune()
        else:
            print("Skipping detuning (use --detune flag to enable)")
        agent.learn()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to load data')
    parser.add_argument('--model_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='Name of the model')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--test_model', type=str, default=None, help='Testing model name')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of episodes to test')
    parser.add_argument('--checkpoint', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='Llama checkpoint name')
    parser.add_argument('--online_eps', type=float, default=0.1, help='Percent of episodes to interact with an online LLM (instead of data)')
    parser.add_argument('--val_type', type=str, choices=['sum', 'mean'], default='sum', help='Type of value function to use')
    parser.add_argument('--prob_granularity', type=str, choices=['sequence', 'token'], default='sequence', help='Granularity of probability distribution')
    parser.add_argument('--alg', type=str, default='ppo', choices=['ppo', 'reinforce', 'ppo-pretrain', 'grpo'], help='Algorithm to use')
    parser.add_argument('--help_weight', type=float, default=0.5, help='Weight for balancing harm vs help advantages (0-1)')
    parser.add_argument('--override', action='store_true', help='Flag to override hyperparameters')
    parser.add_argument('--attack_type', type=str, default='rephrase', choices=['rephrase', 'breakdown'], help='Type of attack to perform during training')
    parser.add_argument('--evaluate', action='store_true', help='Run red-teaming evaluation')
    parser.add_argument('--eval_dataset', type=str, default='PKU-Alignment/PKU-SafeRLHF', help='Dataset to use for evaluation')
    parser.add_argument('--eval_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--detune', action='store_true', help='Flag to enable detuning of model safety alignment')
    # Add arguments for loading fine-tuned models
    parser.add_argument('--use_finetuned', action='store_true', help='Use a fine-tuned model with LoRA adapters')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to the LoRA adapter weights')
    parser.add_argument('--merge_adapters', action='store_true', help='Merge LoRA adapters with the base model for faster inference')
    args = parser.parse_args()
    assert (args.test != args.train) and (args.test or args.train), "Either --test or --train must be specified, but not both"
    #above assertion is just to prevent accidental overwrites 
    torch.autograd.set_detect_anomaly(True)
    main(args=args)