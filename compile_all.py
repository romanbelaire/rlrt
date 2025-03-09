import os
import pandas as pd

def compile_unique_questions(experiment_dir='experiments', output_file='compiled_questions.csv'):
    """
    Parses all experiment directories for 'result_{i}.csv' files and compiles unique questions into one CSV file.

    Args:
        experiment_dir (str): The directory containing experiment subdirectories.
        output_file (str): The path to the output CSV file to save unique questions.
    """
    unique_questions = set()
    compiled_data = []

    # Iterate over all experiment directories
    for experiment in os.listdir(experiment_dir):
        experiment_path = os.path.join(experiment_dir, experiment)
        if os.path.isdir(experiment_path):
            # Iterate over all 'result_{i}.csv' files in the experiment directory
            for file in os.listdir(experiment_path):
                if file.startswith('result_') and file.endswith('.csv'):
                    file_path = os.path.join(experiment_path, file)
                    with open(file_path, 'r') as f:
                        first_char = f.read(1)
                    data = pd.read_csv(file_path, sep=first_char)
                    for question in data['question']:
                        if question not in unique_questions:
                            unique_questions.add(question)
                            compiled_data.append({'question': question})

    # Save the compiled unique questions to the output CSV file
    compiled_df = pd.DataFrame(compiled_data)
    compiled_df.to_csv(output_file, index=False)
    print("Unique questions saved to", output_file)

def compile_successful_questions(experiment_dir='experiments', output_file='successful_questions.csv'):
    """
    Parses all experiment directories for 'result_{i}.csv' files and compiles successful questions into one CSV file.

    A question is considered successful if the score in any of the columns 'gpt_concern', 'gpt_info', 'gpt_naive', or 'gpt_nominal' is 2.
    The corresponding rephrased question from 'rephrase_concern', 'rephrase_info', etc., is added to the output file.

    Args:
        experiment_dir (str): The directory containing experiment subdirectories.
        output_file (str): The path to the output CSV file to save successful questions.
    """
    successful_questions = set()
    compiled_data = []

    # Iterate over all experiment directories
    for experiment in os.listdir(experiment_dir):
        experiment_path = os.path.join(experiment_dir, experiment)
        if os.path.isdir(experiment_path):
            # Iterate over all 'result_{i}.csv' files in the experiment directory
            for file in os.listdir(experiment_path):
                if file.startswith('result_') and file.endswith('.csv'):
                    file_path = os.path.join(experiment_path, file)
                    print(file_path)
                    with open(file_path, 'r') as f:
                        first_char = f.read(1)
                    data = pd.read_csv(file_path, sep=first_char)                    
                    for idx, row in data.iterrows():
                        try:
                            for label, rephrase in [('gpt_concern', 'rephrase_concern'), 
                                                    ('gpt_info', 'rephrase_info'), 
                                                    ('gpt_naive', 'rephrase_naive'), 
                                                    ('gpt_nominal', 'question')]:
                                if row[label] == 2:
                                    question = row[rephrase]
                                    if question not in successful_questions:
                                        successful_questions.add(question)
                                        compiled_data.append({'question': row['question'], 'adversarial': question})
                        except:
                            break

    # Save the compiled successful questions to the output CSV file
    compiled_df = pd.DataFrame(compiled_data)
    compiled_df.to_csv(output_file, index=False)
    print("Successful questions saved to", output_file)

if __name__ == "__main__":
    compile_unique_questions()
    compile_successful_questions() 