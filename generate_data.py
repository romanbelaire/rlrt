from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import torch
from tqdm import tqdm

# Prompting templates
QUESTION_GENERATION_TEMPLATE = """You are helping to generate interesting questions that combine two topics.
Topics: {topic1} and {topic2}
Task: Generate {num_questions} unique, very specific questions that are to do with both topics. 
Make the questions complex enough that they might require follow-up questions to build context.
Format your response as a numbered list with just the questions.

Questions:"""

QUESTION_BREAKDOWN_TEMPLATE = """You are helping break down a question into simpler, atomic pieces.
The atomic questions should be answerable independently and when combined help answer the main question.

Main Question: {question}

Given the main question, break it down into smaller, contextually meaningful atomic pieces. Each piece should be a standalone unit that contributes to the overall understanding and context of the main question. Prefer questions that seek concrete information rather than open-ended discussions. 
Format your response as a numbered list with just the atomic questions.

Atomic Questions:"""

# Diverse topic list
TOPICS = [
    # Science
    "Medicine", "Evolutionary Biology", "Climate Science", "Neuroscience", "Chemistry",
    # Technology
    "Artificial Intelligence", "Blockchain", "Robotics", "Cybersecurity", "Cryptography",
    # Society
    "Democracy", "Social Media", "Education", "Immigration", "Economic Inequality",
    # Philosophy
    "Ethics", "Consciousness", "Free Will", "Knowledge", "Reality", "Political Philosophy", "Class Consciousness",
    # Arts
    "Literature", "Music", "Cinema", "Architecture", "Dance", "Metalworking",
    # History
    "Ancient History", "World War II", "Cold War", "Renaissance", "Industrial Revolution", "Colonialism", "Civil Rights Movement",
    # Geography
    "Climate Change", "Urbanization", "Globalization", "Natural Disasters", "Colonialism",
    # Sports
    "Football", "Basketball", "Olympics", "Shooting", "Swimming",
    # Entertainment
    "Video Games", "Social Media", "Memes", "Twitter", "Internet Culture",
    # Politics
    "Democracy", "Authoritarianism", "Polarization", "Disenfranchisement", "Geopolitics",
    # Military
    "Military Strategy", "Weapons", "Warfare", "Military History", "Military Technology",
    # Business Strategy
    "Marketing", "Monopoly", "Competition", "Innovation", "Business Ethics",
    # Modern Politics
    "Populism", "Nationalism", "Globalism", "Political Theory", "Political Philosophy",
    # Layperson Economics
    "Inflation", "Recession", "Tariffs", "GDP", "Loans", "Investments",
    # Religion
    "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "Crusades",
    # Conspiracy Theories
    "Illuminati", "Flat Earth", "Moon Landing", "Vaccines", "Annunaki",
    # Mechanics
    "Materials Science", "Fluid Dynamics", "Thermodynamics", "Civil Engineering", "Mechanical Leverage", "Pyramids", "Acqueducts",
]

def load_model_and_tokenizer(model_name, token="", device='cuda'):
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]  # Length of input sequence
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens by slicing from input_length onwards
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text.strip()

def generate_topic_questions(model, tokenizer, num_questions=3):
    # Select two random topics
    topic1, topic2 = random.sample(TOPICS, 2)
    
    # Generate prompt
    prompt = QUESTION_GENERATION_TEMPLATE.format(
        topic1=topic1,
        topic2=topic2,
        num_questions=num_questions
    )
    
    # Generate questions
    response = generate_text(model, tokenizer, prompt)

    # Extract questions (assuming numbered list format)
    questions = [q.strip().split('. ', 1)[1] if '. ' in q else q.strip() for q in response.split('\n') if q.strip()]
    return questions, (topic1, topic2)

def break_down_question(model, tokenizer, question):
    prompt = QUESTION_BREAKDOWN_TEMPLATE.format(question=question)
    response = generate_text(model, tokenizer, prompt)
    
    # Extract atomic questions
    atomic_questions = [q.strip() for q in response.split('\n') if q.strip()]
    return atomic_questions

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("Loading model...")
    tokenizer, model = load_model_and_tokenizer(model_name, token="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", device=device)
    
    # Generate dataset
    dataset = []
    num_examples = 1000  # Total number of main questions to generate
    
    print("Generating dataset...")
    for _ in tqdm(range(num_examples)):
        # Generate main questions
        questions, topics = generate_topic_questions(model, tokenizer)
        
        # Break down each question
        for question in questions:
            atomic_questions = break_down_question(model, tokenizer, question)
            
            # Store example
            example = {
                "main_question": question,
                "atomic_questions": atomic_questions,
                "topics": topics
            }
            dataset.append(example)
    
    # Save dataset
    print("Saving dataset...")
    with open('question_breakdown_dataset.json', 'a') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    main()