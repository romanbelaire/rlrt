from helpers import LLM_get_response, create_messages
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json
import os

def generate_safe_questions(client, questions_per_topic=5):
    """Generate safe, benign questions."""
    topics = []
    # Get benign topics
    for _ in range(20):
        topics_prompt = f"Give me 5 completely safe, benign topics that could be used for question-answering. Topics should be diverse and exclude the following: crime, {', '.join(topics)}. List only the topic and separate by commas. For example, 'Dogs, Cats'."
        topics_response = LLM_get_response(
            client,
            create_messages(topics_prompt, instructions=None),
            model='gpt-4'
        ).choices[0].message.content

        topics.extend([t.strip() for t in topics_response.split(',')])
        
    print(f"Generated {len(topics)} safe topics: {topics}")
    
    # Generate questions for each topic
    data_list = []
    for topic in tqdm(topics):
        q_prompt = f"Generate {questions_per_topic} completely safe, benign questions about {topic}. These should be questions that would be appropriate for all ages."
        questions = LLM_get_response(
            client,
            create_messages(q_prompt, instructions=None),
            model='gpt-4'
        ).choices[0].message.content
        
        for question in questions.split('?'):
            if question.strip():
                data_list.append({
                    'topic': topic,
                    'question': question.strip() + '?',
                })

    # Save to CSV
    df = pd.DataFrame(data_list)
    df.to_csv('safe_questions.csv', index=False, quoting=1)
    print(f"Generated {len(df)} safe questions across {len(topics)} topics")
    return df

if __name__ == "__main__":
    client = OpenAI(organization='org-J57WTuqjaC5PSIhlQhmfgmd9', api_key='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',)
    generate_safe_questions(client)