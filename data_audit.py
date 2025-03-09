import os
import pandas as pd
import random

def audit_dataset(experiment_dir='experiments'):
    all_questions = []
    
    # Iterate over all experiment directories
    for experiment in os.listdir(experiment_dir):
        experiment_path = os.path.join(experiment_dir, experiment)
        if os.path.isdir(experiment_path):
            # Iterate over all 'result_{i}.csv' files in the experiment directory
            for file in os.listdir(experiment_path):
                if file.startswith('result_') and file.endswith('.csv'):
                    file_path = os.path.join(experiment_path, file)
                    try:
                        data = pd.read_csv(file_path)
                    except:
                        data = pd.read_csv(file_path, sep='§', engine='python')
                    all_questions.extend(data['question'].tolist())
    
    num_entries = len(all_questions)
    print(f"Number of entries: {num_entries}")
    
    # Print 2% of the dataset questions
    sample_size = max(1, int(0.02 * num_entries))
    sample_questions = random.sample(all_questions, sample_size)
    
    for question in sample_questions:
        print(question)
        print("\n")

    return all_questions

def clean_dataset(ds):
    cleaned_questions = []

    # First pass: split questions with multiple '?' and remove the original
    for question in ds:
        if isinstance(question, str):
            if question.count('?') > 1:
                split_questions = [q.strip() + '?' for q in question.split('?') if q.strip()]
                cleaned_questions.extend(split_questions)
            else:
                cleaned_questions.append(question)

    # Second pass: ensure each question ends with '?' and remove invalid questions
    final_questions = []
    for question in cleaned_questions:
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        if question.strip() != 'nan' and "i can't help" not in question.lower() and "i can't assist" not in question.lower():
            final_questions.append(question)

    for question in final_questions:
        if question.startswith(','):
            question = question.lstrip(',').strip()
            
    # Remove duplicate questions
    final_questions = list(dict.fromkeys(final_questions))

    return final_questions

# Run the audit
ds = audit_dataset()
cleaned_ds = clean_dataset(ds)

df = pd.DataFrame(cleaned_ds, columns=['questions'])
df.to_csv('cleaned_questions.csv', index=False, quoting=1)