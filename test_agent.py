import csv
from datetime import datetime
import os
import pandas as pd
import glob
import json
import torch
import numpy as np
from agent import Agent
import sys


class TestAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, dataset_name, num_samples=100, batch_size=8, adapter_types=None):
        """
        Test the agent in the given environment.

        Args:
            dataset_name (str): HuggingFace dataset name to use for evaluation
            num_samples (int): Number of samples to evaluate
            batch_size (int): Batch size for processing
            adapter_types (list, optional): List of adapter types to test with. If provided,
                the method will run the evaluation for each adapter type separately.
                Example: ['lm_head_rephrase', 'attention_rephrase', 'lm_head_followup', 'attention_followup']

        Returns:
            dict: Dictionary containing evaluation metrics for each adapter type if adapter_types is provided,
                otherwise metrics for the current model configuration.
        """
        if not adapter_types or not self.use_lora:
            # Run evaluation with current model configuration
            results = self.evaluate_red_teaming(dataset_name, num_samples=num_samples, batch_size=batch_size)
            print(results)
            return results
        
        # Test with each adapter type separately
        all_results = {}
        original_adapter = self.current_adapter
        
        try:
            for adapter_type in adapter_types:
                print(f"\n--- Testing with adapter: {adapter_type} ---")
                
                # Switch to the adapter
                success = self.switch_lora_adapter(adapter_type)
                
                if not success:
                    print(f"Failed to switch to adapter {adapter_type}, skipping...")
                    all_results[adapter_type] = {"error": "Failed to load adapter"}
                    continue
                
                # Run evaluation
                results = self.evaluate_red_teaming(dataset_name, num_samples=num_samples, batch_size=batch_size)
                all_results[adapter_type] = results
                print(f"Results for {adapter_type}:")
                print(results)
        finally:
            # Switch back to the original adapter
            if original_adapter:
                self.switch_lora_adapter(original_adapter)
        
        return all_results

    def load_harmbench_data(self, num_samples=None):
        """
        Load HarmBench data from CSV files in the HarmBench directory.
        
        Args:
            num_samples (int, optional): Number of samples to load. If None, load all samples.
            
        Returns:
            pd.DataFrame: DataFrame containing HarmBench data with Behavior and ContextString combined
        """
        print("Loading HarmBench data...")
        
        # Find all CSV files in the HarmBench directory that don't have "test" or "val" in the filename
        csv_files = glob.glob("HarmBench/*.csv")
        csv_files = [f for f in csv_files if "test" not in f.lower() and "val" not in f.lower()]
        
        if not csv_files:
            raise FileNotFoundError("No suitable CSV files found in HarmBench directory")
        
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        # Load and combine all dataframes
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                print(f"Loaded {file} with {len(df)} rows")
                
                # Make sure the required columns exist
                if "Behavior" not in df.columns:
                    print(f"Warning: 'Behavior' column not found in {file}, skipping")
                    continue
                    
                # If ContextString exists, append it to Behavior
                if "ContextString" in df.columns:
                    # Create a new column for the combined behavior and context
                    df['CombinedBehavior'] = df['Behavior'].copy()
                    
                    # Count how many contexts we're processing
                    context_count = 0
                    
                    # Only combine if ContextString is not empty/null
                    for i, row in df.iterrows():
                        if pd.notna(row['ContextString']) and row['ContextString'].strip():
                            context = row['ContextString'].strip()
                            # Remove surrounding quotes if they exist
                            if (context.startswith('"') and context.endswith('"')) or \
                               (context.startswith("'") and context.endswith("'")):
                                context = context[1:-1]
                            
                            # Add a space between behavior and context if needed
                            if not row['CombinedBehavior'].endswith(' '):
                                df.at[i, 'CombinedBehavior'] = f"{row['CombinedBehavior']} {context}"
                            else:
                                df.at[i, 'CombinedBehavior'] = f"{row['CombinedBehavior']}{context}"
                            context_count += 1
                    
                    print(f"  - Processed {context_count} context strings in {file}")
                    
                    # Use the combined column as the behavior
                    df['Behavior'] = df['CombinedBehavior']
                    df = df.drop('CombinedBehavior', axis=1)
                
                # Sample check - print a few examples
                if len(df) > 0:
                    sample_idx = min(3, len(df))
                    print(f"  - Sample behaviors from {file}:")
                    for i in range(sample_idx):
                        # Print first 100 chars of each behavior
                        behavior = df.iloc[i]['Behavior']
                        print(f"    {i+1}. {behavior[:100]}{'...' if len(behavior) > 100 else ''}")
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dfs:
            raise ValueError("No valid data found in HarmBench CSV files")
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Limit to the requested number of samples if specified
        if num_samples is not None and num_samples < len(combined_df):
            combined_df = combined_df.sample(num_samples, random_state=42)
            print(f"Randomly sampled {num_samples} examples from {len(combined_df)} total")
        
        print(f"Loaded {len(combined_df)} total samples from HarmBench")
        return combined_df

    def evaluate_red_teaming(self, dataset_name, split="train", num_samples=10, batch_size=1):
        """
        Evaluates the effectiveness of the red-teaming model by comparing harmfulness
        of target LLM responses to original prompts vs. red-teamed prompts in a chat setting.

        Args:
            dataset_name (str): HuggingFace dataset name or path to use for evaluation
            split (str): Dataset split to use (default: "train")
            num_samples (int): Number of samples to evaluate (default: 100)
            batch_size (int): Batch size for processing (default: 8)

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        from tqdm import tqdm
        import traceback

        print(f"Starting red-teaming evaluation...")
        self.env.end_criteria['target_refusal'] = True
        
        # Flag to track if we're using HarmBench data
        using_harmbench = False
        
        # Check if we're using HarmBench data
        if "harmbench" in dataset_name.lower():
            using_harmbench = True
            try:
                # Load HarmBench data
                dataset_df = self.load_harmbench_data(num_samples)
                prompts = dataset_df['Behavior'].tolist()
                print(f"Using HarmBench data with {len(prompts)} prompts")
            except Exception as e:
                print(f"Error loading HarmBench data: {e}")
                traceback.print_exc()
                return {"error": str(e)}
        else:
            # Load dataset from HuggingFace
            from datasets import load_dataset
            print(f"Loading dataset {dataset_name} split {split}")
            dataset = load_dataset(dataset_name, split=split)
            if len(dataset) > num_samples:
                dataset = dataset.select(range(num_samples))

            # Determine prompt field based on dataset structure
            prompt_field = self._detect_prompt_field(dataset[0])
            print(f"Using '{prompt_field}' as the prompt field")
            
            # Extract prompts
            prompts = dataset[prompt_field]

        # Initialize results storage
        results = {
            'original_prompts': [],
            'red_teamed_prompts': [],
            'original_responses': [],
            'red_teamed_responses': [],
            'original_harm_scores': [],
            'red_teamed_harm_scores': [],
            'original_refusals': [],
            'red_teamed_refusals': [],
            'red_teamed_chat_scores': [],
            'episode_lengths': [],
            'cumulative_rewards': [],  # Track total rewards for each episode
            'step_rewards': [],        # Track rewards at each step of conversation
            'episode_ids': []          # Track unique episode IDs
        }
        
        # Process in batches
        try:
            episode_id = 0
            for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating batches"):
                batch = prompts[i:i+batch_size]
                # Skip incomplete batches to avoid tensor size mismatches
                if len(batch) != batch_size:
                    print(f"Skipping incomplete batch of size {len(batch)} (expected {batch_size})")
                    continue
                
                # Set batch size for environment
                self.env.set_batch_size(batch_size)
                
                original_prompts = batch
                
                # Step 1: Get baseline responses and scores using the environment
                original_responses = []
                original_harm_scores = []
                original_refusals = []
                
                # Reset environment with the original prompt
                self.env.reset()
                self.env.set_state(original_prompts)
                
                # Get response from target model
                chat_history = [q[0] for q in self.env.state]
                assert isinstance(chat_history, list) and isinstance(chat_history[0], str), f"chat_history is not a list of strings: {chat_history}"
                
                state, responses, rewards, done = self.env.step(chat_history)
                
                original_responses.extend(responses)
                
                # Score the response
                harm_score = self.env.score_safety(responses, questions=None, include_white_box=False).cpu().numpy()
                original_harm_scores.extend(harm_score)
                
                # Check for refusals
                refusal = self.env.check_refusals(responses).cpu().numpy()
                original_refusals.extend(refusal)
                
                # Step 2: Generate red-teamed versions of the prompts
                red_teamed_prompts = []
                
                # Step 3: Get responses to red-teamed prompts
                red_teamed_responses = []
                red_teamed_harm_scores = []
                red_teamed_refusals = []
                red_teamed_chat_scores = []
                episode_lengths = []
                
                # Initialize batch reward tracking
                batch_cumulative_rewards = [0.0] * batch_size
                batch_step_rewards = [[] for _ in range(batch_size)]
                batch_episode_ids = [episode_id + j for j in range(batch_size)]
                
                # Reset environment with the original prompt
                self.env.reset()
                self.env.set_state(original_prompts)
                state = self.env.state
                
                # Step through the environment until done
                step = 0
                while step < 3:
                    self.to_cuda()
                    action = self.act(state, self.env.base_state, self.instructions, enforce_single_question=True)[0]
                    self.to_cpu()
                    
                    if step == 0:
                        if isinstance(action, str):
                            red_teamed_prompts.append(action)
                        else:
                            red_teamed_prompts.extend(action)
                    else:
                        for j in range(batch_size):
                            if not done[j]:
                                if isinstance(action, str):
                                    red_teamed_prompts[j] += '[sep]' + action
                                else:
                                    red_teamed_prompts[j] += '[sep]' + action[j]
                    
                    # Get response from target model using the red-teamed prompt
                    if not isinstance(action, list):
                        action = [action] * batch_size  # Ensure action is a list matching batch size
                    
                    state, responses, rewards, done = self.env.step(action)
                    
                    # Track rewards at each step
                    for j in range(batch_size):
                        if not done[j]:
                            # Record the reward for this step
                            reward_value = rewards[j].item() if isinstance(rewards[j], torch.Tensor) else rewards[j]
                            batch_step_rewards[j].append(reward_value)
                            batch_cumulative_rewards[j] += reward_value
                    
                    step += 1
                    del action

                    if all(done):
                        break
                
                # Get the final response for evaluation
                red_teamed_responses.extend(responses)
                
                # Score the response
                harm_score = self.env.score_safety(responses, questions=None, include_white_box=False).cpu().numpy()
                red_teamed_harm_scores.extend(harm_score)
                
                # Scores the chat history
                hist_str = ["" for _ in range(batch_size)]
                for j, r in enumerate(responses):
                    hist_str[j] += r[0] 
                chat_score = self.env.score_safety(hist_str, questions=None, include_white_box=False).cpu().numpy()
                red_teamed_chat_scores.extend(chat_score)
                
                # Check for refusals
                refusal = self.env.check_refusals(responses).cpu().numpy()
                red_teamed_refusals.extend(refusal)

                # Store episode lengths
                current_episode_lengths = [self.env.current_step] * batch_size
                episode_lengths.extend(current_episode_lengths)

                # Store results
                results['original_prompts'].extend(original_prompts)
                results['red_teamed_prompts'].extend(red_teamed_prompts)
                results['original_responses'].extend(original_responses)
                results['red_teamed_responses'].extend(red_teamed_responses)
                results['original_harm_scores'].extend(original_harm_scores)
                results['red_teamed_harm_scores'].extend(red_teamed_harm_scores)
                results['original_refusals'].extend(original_refusals)
                results['red_teamed_refusals'].extend(red_teamed_refusals)
                results['red_teamed_chat_scores'].extend(red_teamed_chat_scores)
                results['episode_lengths'].extend(current_episode_lengths)
                results['cumulative_rewards'].extend(batch_cumulative_rewards)
                results['step_rewards'].extend(batch_step_rewards)
                results['episode_ids'].extend(batch_episode_ids)
                
                # Update episode ID counter
                episode_id += batch_size
                
                print(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            traceback.print_exc()
            return {"error": str(e), "partial_results": results}
            
        # Calculate metrics
        metrics = self._calculate_red_teaming_metrics(results)
        
        # Add metrics to results
        results.update(metrics)
        
        # Save intermediate results
        self._save_intermediate_results(results)
        
        # Save top episodes
        self._save_top_episodes(results)
        
        # Generate correlation plots if matplotlib is available
        if 'matplotlib' in sys.modules:
            try:
                self._save_correlation_plots(results)
            except Exception as e:
                print(f"Error creating correlation plots: {str(e)}")
        
        # If using HarmBench data, save specific HarmBench results
        if using_harmbench:
            try:
                results_path = self.save_harmbench_results(results, output_dir=f"./models/{self.model_name}/harmbench_results")
                print(f"Saved HarmBench results to {results_path}")
            except Exception as e:
                print(f"Error saving HarmBench results: {str(e)}")
                traceback.print_exc()
        
        return results

    def _save_intermediate_results(self, results):
        """Save intermediate results during evaluation to prevent data loss"""
        for k, v in results.items():
            print(f"{k}: {len(v)}")
            if isinstance(v[0], str):
                v = ["\"" + str(item) + "\"" for item in v]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"./models/{self.model_name}", exist_ok=True)
        
        # Save to CSV with timestamp
        intermediate_file = f"./models/{self.model_name}/red_teaming_results_intermediate_{timestamp}.csv"
        
        # Convert step_rewards to strings for CSV
        results_for_csv = results.copy()
        results_for_csv['step_rewards'] = [json.dumps(r) for r in results['step_rewards']]
        
        pd.DataFrame(results_for_csv).to_csv(intermediate_file, index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        print(f"Intermediate results saved at {timestamp}")

    def _save_top_episodes(self, results):
        """Save the top 20 highest scoring episodes to a JSON file"""
        try:
            # Create a score based on reward and episode length
            episode_data = []
            
            for i in range(len(results['episode_ids'])):
                episode_info = {
                    'episode_id': results['episode_ids'][i],
                    'original_prompt': results['original_prompts'][i],
                    'red_teamed_prompt': results['red_teamed_prompts'][i],
                    'original_response': results['original_responses'][i],
                    'red_teamed_response': results['red_teamed_responses'][i],
                    'original_harm_score': float(results['original_harm_scores'][i]),
                    'red_teamed_harm_score': float(results['red_teamed_harm_scores'][i]),
                    'original_refusal': bool(results['original_refusals'][i]),
                    'red_teamed_refusal': bool(results['red_teamed_refusals'][i]),
                    'chat_score': float(results['red_teamed_chat_scores'][i]),
                    'episode_length': results['episode_lengths'][i],
                    'cumulative_reward': float(results['cumulative_rewards'][i]),
                    'step_rewards': results['step_rewards'][i]
                }
                episode_data.append(episode_info)
            
            # Sort episodes by cumulative reward (highest first)
            sorted_episodes = sorted(episode_data, key=lambda x: x['cumulative_reward'], reverse=True)
            
            # Get top 20 episodes
            top_20_episodes = sorted_episodes[:20]
            
            # Save to JSON
            output_file = f"./models/{self.model_name}/top_20_episodes.json"
            with open(output_file, 'w') as f:
                json.dump(top_20_episodes, f, indent=2)
            
            print(f"Top 20 episodes saved to {output_file}")
        except Exception as e:
            print(f"Error saving top episodes: {str(e)}")

    def _detect_prompt_field(self, example):
        """Detect which field in the dataset contains the prompt"""
        potential_fields = ['prompt', 'question', 'input', 'text', 'content']
        for field in potential_fields:
            if field in example:
                return field
        
        # If no standard field is found, use the first string field
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 10:
                return key
        
        raise ValueError("Could not detect a suitable prompt field in the dataset")

    def generate_red_teamed_prompt(self, original_prompt):
        """Generate a red-teamed version of the original prompt using the actor model for"""
        # Prepare system prompt for red-teaming
        with torch.no_grad():
            if self.attack_type == 'rephrase':
                outputs, _, _ = self.act(original_prompt, self.env.base_state, self.instructions, enforce_single_question=True)
            else:  # breakdown
                outputs, _, _ = self.act_secret(original_prompt, self.env.base_state, self.instructions, enforce_single_question=True)
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        red_teamed_prompt = full_response.split("assistant")[-1].strip()
        assert len(red_teamed_prompt) > 0, f"Red-teamed prompt is empty: {red_teamed_prompt}"
        
        # Return to CPU to save memory
        self.actor.model.to('cpu')
        
        return red_teamed_prompt

    def _calculate_red_teaming_metrics(self, results):
        """Calculate metrics based on evaluation results"""
        import numpy as np
        
        # Convert to numpy arrays for easier calculation
        original_harm = np.array(results['original_harm_scores'])
        red_teamed_harm = np.array(results['red_teamed_harm_scores'])
        original_refusals = np.array(results['original_refusals'])
        red_teamed_refusals = np.array(results['red_teamed_refusals'])
        episode_lengths = np.array(results['episode_lengths'])
        cumulative_rewards = np.array(results['cumulative_rewards'])
        
        # Calculate metrics
        harm_increased = red_teamed_harm > original_harm
        harm_increased = np.array(harm_increased)
        harm_increase_rate = harm_increased.mean() * 100
        
        # Average harm score delta
        harm_delta = red_teamed_harm - original_harm
        harm_delta = np.array(harm_delta)
        avg_harm_delta = harm_delta.mean()
        
        # Refusal rates
        original_refusal_rate = original_refusals.mean() * 100
        red_teamed_refusal_rate = red_teamed_refusals.mean() * 100
        refusal_rate_change = red_teamed_refusal_rate - original_refusal_rate
        
        # Success metrics
        success_rate = ((harm_increased) & (~red_teamed_refusals)).mean() * 100
        
        # Analyze correlation between episode length and metrics
        length_reward_correlation = np.corrcoef(episode_lengths, cumulative_rewards)[0, 1]
        
        # Group by episode length and analyze refusal rates and rewards
        unique_lengths = np.unique(episode_lengths)
        length_metrics = {}
        
        for length in unique_lengths:
            indices = episode_lengths == length
            length_metrics[int(length)] = {
                'count': int(indices.sum()),
                'avg_reward': float(cumulative_rewards[indices].mean()),
                'refusal_rate': float(red_teamed_refusals[indices].mean() * 100),
                'harm_score': float(red_teamed_harm[indices].mean()),
                'success_rate': float(((harm_increased[indices]) & (~red_teamed_refusals[indices])).mean() * 100)
            }

        # Compile metrics
        metrics = {
            'harm_increase_rate': float(harm_increase_rate),
            'avg_harm_delta': float(avg_harm_delta),
            'original_refusal_rate': float(original_refusal_rate),
            'red_teamed_refusal_rate': float(red_teamed_refusal_rate),
            'refusal_rate_change': float(refusal_rate_change),
            'success_rate': float(success_rate),
            'num_samples': len(original_harm),
            'red_teamed_chat_scores': float(np.array(results['red_teamed_chat_scores']).mean()),
            'length_reward_correlation': float(length_reward_correlation) if not np.isnan(length_reward_correlation) else 0,
            'length_metrics': length_metrics
        }
        
        # Save summary metrics
        with open(f"./models/{self.model_name}/red_teaming_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Create and save a correlation analysis plot
        self._save_correlation_plots(results)

        return metrics
        
    def _save_correlation_plots(self, results):
        """Create and save plots showing correlations between episode length, refusal rates, and rewards"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            episode_lengths = np.array(results['episode_lengths'])
            cumulative_rewards = np.array(results['cumulative_rewards'])
            red_teamed_refusals = np.array(results['red_teamed_refusals'])
            
            # Create output directory
            plot_dir = f"./models/{self.model_name}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot 1: Episode Length vs Reward
            plt.figure(figsize=(10, 6))
            plt.scatter(episode_lengths, cumulative_rewards, alpha=0.6)
            plt.title('Episode Length vs Cumulative Reward')
            plt.xlabel('Episode Length')
            plt.ylabel('Cumulative Reward')
            
            # Add regression line
            z = np.polyfit(episode_lengths, cumulative_rewards, 1)
            p = np.poly1d(z)
            plt.plot(sorted(episode_lengths), p(sorted(episode_lengths)), "r--", alpha=0.8)
            
            # Add correlation coefficient to the plot
            corr = np.corrcoef(episode_lengths, cumulative_rewards)[0, 1]
            plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/length_vs_reward.png")
            plt.close()
            
            # Plot 2: Average reward by episode length
            unique_lengths = np.unique(episode_lengths)
            avg_rewards = []
            refusal_rates = []
            
            for length in unique_lengths:
                indices = episode_lengths == length
                avg_rewards.append(cumulative_rewards[indices].mean())
                refusal_rates.append(red_teamed_refusals[indices].mean() * 100)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Episode Length')
            ax1.set_ylabel('Average Reward', color='tab:blue')
            ax1.plot(unique_lengths, avg_rewards, 'o-', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Refusal Rate (%)', color='tab:red')
            ax2.plot(unique_lengths, refusal_rates, 'o-', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title('Average Reward and Refusal Rate by Episode Length')
            fig.tight_layout()
            plt.savefig(f"{plot_dir}/avg_metrics_by_length.png")
            plt.close()
            
            print(f"Correlation plots saved to {plot_dir}/")
        except Exception as e:
            print(f"Error creating correlation plots: {str(e)}")
            # Continue execution even if plotting fails

    def save_harmbench_results(self, results, output_dir="output/harmbench_results"):
        """
        Save HarmBench testing results to CSV files.
        
        Args:
            results (dict): Dictionary containing test results
            output_dir (str): Directory to save results
            
        Returns:
            str: Path to saved results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results dataframe
        results_df = pd.DataFrame({
            'original_prompt': results['original_prompts'],
            'red_teamed_prompt': results['red_teamed_prompts'],
            'original_response': results['original_responses'],
            'red_teamed_response': results['red_teamed_responses'],
            'original_harm_score': results['original_harm_scores'],
            'red_teamed_harm_score': results['red_teamed_harm_scores'],
            'original_refusal': results['original_refusals'],
            'red_teamed_refusal': results['red_teamed_refusals'],
            'episode_length': results['episode_lengths'],
            'cumulative_reward': results['cumulative_rewards']
        })
        
        # Save to CSV
        results_path = os.path.join(output_dir, f"harmbench_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")
        
        # Save metrics summary
        metrics = self._calculate_red_teaming_metrics(results)
        metrics_path = os.path.join(output_dir, f"harmbench_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")
        
        return results_path
