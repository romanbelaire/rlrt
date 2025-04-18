import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import traceback
from agent import Agent, test_inputs
from test_agent import TestAgent
import argparse
import torch 
from datetime import datetime
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from base import BaseModelSingleton
from BTagent import BTagent
# Import AccelAgent and AccelBTAgent
from AccelAgent import AccelAgent, AccelBTAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Epsilon for PPO')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--minibatch_size', type=int, default=16, help='Minibatch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--max_ep_len', type=int, default=4096, help='Maximum episode length')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--value_lr', type=float, default=5e-3, help='Value function learning rate')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to load data')
    parser.add_argument('--model_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='Name of the model')
    parser.add_argument('--test', action='store_true', help='Flag to indicate testing mode')
    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--test_model', type=str, default=None, help='Testing model name')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of episodes to test')
    parser.add_argument('--checkpoint', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='Llama checkpoint name')
    parser.add_argument('--eps', type=float, default=0.1, help='exploration rate')
    parser.add_argument('--ce_coef', type=float, default=0.5, help='Coefficient for cross-entropy language modeling loss')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Coefficient for value function loss')
    parser.add_argument('--sem_coef', type=float, default=0.5, help='Coefficient for semantic similarity loss')
    parser.add_argument('--adv_coef', type=float, default=0.1, help='Coefficient for adversarial loss (not used)')
    parser.add_argument('--kl_coef', type=float, default=0.1, help='Coefficient for KL divergence loss')
    parser.add_argument('--val_type', type=str, choices=['sum', 'mean'], default='sum', help='Type of value function to use')
    parser.add_argument('--prob_granularity', type=str, choices=['sequence', 'token'], default='sequence', help='Granularity of probability distribution')
    parser.add_argument('--alg', type=str, default='ppo', choices=['ppo', 'reinforce', 'ppo-pretrain', 'grpo'], help='Algorithm to use')
    parser.add_argument('--help_weight', type=float, default=0.5, help='Weight for balancing harm vs help advantages (0-1)')
    parser.add_argument('--override', action='store_true', help='Flag to override hyperparameters')
    parser.add_argument('--attack_type', type=str, default='rephrase', choices=['rephrase', 'breakdown'], help='Type of attack to perform during training')
    parser.add_argument('--eval_dataset', type=str, default='PKU-Alignment/PKU-SafeRLHF', help='Dataset to use for evaluation')
    parser.add_argument('--eval_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--detune', action='store_true', help='Flag to enable detuning of model safety alignment')
    parser.add_argument('--recompute_advantages', action='store_true', help='Flag to recompute advantages instead of using GAE')
    parser.add_argument('--sum_values_into_action', type=bool, default=True, help='Flag to sum values into the action dimension')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving the model')
    # Add arguments for loading fine-tuned models
    parser.add_argument('--use_finetuned', action='store_true', help='Use a fine-tuned model with LoRA adapters')
    parser.add_argument('--use_detuned', action='store_true', help='Use a detuned model')
    parser.add_argument('--base_adapter_path', type=str, default=None, help='Path to the base model LoRA adapter weights')
    parser.add_argument('--merge_adapters', action='store_true', help='Merge LoRA adapters with the base model for faster inference')
    parser.add_argument('--target_model', type=str, default=None, help='Name of the target model to use for evaluation (if different from base model)')
    parser.add_argument('--keep_base_on_cpu', action='store_true', help='Keep the base model on CPU')
    # Add LoRA fine-tuning arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA adapters')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout parameter')
    # Add BTagent arguments
    parser.add_argument('--use_btagent', action='store_true', help='Use templated BTagent')
    parser.add_argument('--top_tokens_count', type=int, default=5, help='Number of top tokens to use for BTagent')
    
    # Add adapter type argument
    parser.add_argument('--adapter_type_id', type=int, default=None, choices=[0, 1, 2, 3, 4], 
                        help='ID of the adapter type to use: 0=lm_head_rephrase, 1=attention_rephrase, 2=lm_head_followup, 3=attention_followup, 4=attention_holistic')
    
    # Add target adapter path argument
    parser.add_argument('--target_adapter_path', type=str, default=None, help='Path to the target model LoRA adapter weights')
    
    # Custom reward model arguments
    parser.add_argument('--custom_harm_critic', type=str, default=None,
                      help='Path to custom harm critic model checkpoint')
    parser.add_argument('--custom_env_reward', type=str, default=None,
                      help='Path to custom environment reward model checkpoint')
    parser.add_argument('--use_custom_reward', action='store_true',
                      help='Use custom reward model in environment')
    
    return parser.parse_args()

def main():
    args = parse_args()
    # Create auth token
    with open("hf_api", "r") as f:
        auth_token = f.read().strip()
    
    # Initialize separate model singletons
    agent_model_singleton = BaseModelSingleton.get_instance(
        args.checkpoint, 
        hf_token=auth_token, 
        keep_base_on_cpu=args.keep_base_on_cpu
    )
    
    # Value critics and environment target always use Llama-3.2-1B-Instruct
    llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Create singleton instances - critic and target models share the same model type
    critic_model_singleton = BaseModelSingleton.get_instance(
        llama_model_name, 
        hf_token=auth_token, 
        keep_base_on_cpu=args.keep_base_on_cpu
    )
    
    # If a specific target model is specified, use that instead of the default
    target_model_name = args.target_model if args.target_model else llama_model_name
    if target_model_name != llama_model_name:
        target_model_singleton = BaseModelSingleton.get_instance(
            target_model_name, 
            hf_token=auth_token, 
            keep_base_on_cpu=args.keep_base_on_cpu
        )
    else:
        target_model_singleton = critic_model_singleton
    hyperparameters = args.__dict__
    
    # Define a mapping from adapter type ID to adapter type name
    adapter_type_mapping = {
        0: "lm_head_rephrase",
        1: "attention_rephrase", 
        2: "lm_head_followup",
        3: "attention_followup",
        4: "attention_holistic"
    }
    
    # Get the adapter type name based on the ID
    if args.adapter_type_id is not None:
        adapter_type = adapter_type_mapping[args.adapter_type_id]
        print(f"Using adapter type: {adapter_type} (ID: {args.adapter_type_id})")
    else:
        adapter_type = None
    
    # If adapter_path is specified, ensure it's an absolute path
    if args.base_adapter_path:
        if not os.path.isabs(args.base_adapter_path):
            args.base_adapter_path = os.path.abspath(args.base_adapter_path)
        print(f"Using base adapter path: {args.base_adapter_path}")
        
        # Check if adapter_path exists
        if not os.path.exists(args.base_adapter_path):
            print(f"Warning: Base adapter path {args.base_adapter_path} does not exist!")
    
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

    policy_type = 'CPUTransformer' if args.keep_base_on_cpu else 'Transformer'

    # Create Agent with optional fine-tuned model
    # Determine Agent Type based on flags and model size
    if "70b" in args.checkpoint.lower():
        print("70b model detected, using Accelerated Agent.")
        if args.use_btagent:
            print("Using AccelBTAgent.")
            AgentType = AccelBTAgent
        else:
            print("Using AccelAgent.")
            AgentType = AccelAgent
    else:
        # Original logic for non-70b models
        AgentType = Agent if args.train else TestAgent
        AgentType = BTagent if args.use_btagent else AgentType
        print(f"Using Agent Type: {AgentType.__name__}")
    
    # Create the agent with the environment
    agent = AgentType(
        policy_type=policy_type, 
        data_path=args.data_dir, 
        hf_checkpoint=args.checkpoint, 
        hyperparameters=hyperparameters, 
        auth_token=auth_token,
        use_finetuned=args.use_finetuned,
        base_adapter_path=args.base_adapter_path,
        merge_adapters=args.merge_adapters,
        use_lora=args.use_lora,
        adapter_type=adapter_type,
        agent_model_singleton=agent_model_singleton,
        critic_model_singleton=critic_model_singleton,
        target_model_singleton=None,
        target_adapter_path=args.target_adapter_path,
        custom_harm_critic_path=args.custom_harm_critic
    )
    
    # Pass custom environment parameters to the agent's environment
    if args.use_custom_reward and args.custom_env_reward:
        agent.env.use_custom_reward = args.use_custom_reward
        agent.env.reward_model_path = args.custom_env_reward
        success = agent.env.load_custom_reward_model(args.custom_env_reward)
        if success:
            agent.env.score_safety = agent.env.custom_score_safety

    if args.test:
        print(f"Evaluating red-teaming effectiveness using {args.eval_dataset}")
        
        # If using LoRA and test_adapters is specified, test with each adapter type
        if args.use_lora:
            if adapter_type is None:
                print("No adapter type specified with --use_lora")
                adapter_type = ['lm_head_rephrase', 'attention_rephrase', 'lm_head_followup', 'attention_followup', 'attention_holistic']
            else:
                adapter_type = [adapter_type]
            print(f"Testing with the following adapters: {adapter_type}")
            metrics = agent.test(
                dataset_name=args.eval_dataset,
                num_samples=args.eval_samples,
                batch_size=8,
                adapter_types=adapter_type
            )
            print("Evaluation metrics for each adapter:")
            for adapter_type, adapter_metrics in metrics.items():
                print(f"\n=== Results for {adapter_type} ===")
                for key, value in adapter_metrics.items():
                    print(f"  {key}: {value}")
        else:
            # Standard testing without multiple adapters or with single adapter specified by adapter_type
            metrics = agent.test(
                dataset_name=args.eval_dataset,
                num_samples=args.eval_samples,
                batch_size=4,
                adapter_types= None
            )
            print("Evaluation metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
    else:
        print("Training agent with algorithm: ", args.alg)
        if args.detune:
            print("Detuning model safety alignment...")
            agent.detune()
        else:
            print("Skipping detuning (use --detune flag to enable)")
        agent.learn()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    start_time = datetime.now()
    try:
        main()
    except Exception as e:
        print(f"Program terminated after {datetime.now() - start_time}")
        print(e)
        print("Traceback: ", traceback.format_exc())
        raise e
