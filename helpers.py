import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def LLM_get_response(oai_client, prompt, num_responses=1, print_output=False, model='gpt-3.5-turbo', max_tokens=100):
    """
    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    completions = oai_client.chat.completions.create(
        model=model,#'gpt-3.5-turbo',  # Determines the quality, speed, and cost.
        temperature=0.5,            # Level of creativity in the response
        messages=prompt,           # What the user typed in
        max_tokens=max_tokens,             # Maximum tokens in the prompt AND response
        #n=num_responses,                        # The number of completions to generate
        #stop=None,                  # An optional setting to control response generation
    )

    # Displaying the output can be helpful if things go wrong
    if print_output:
        print(completions)

    # Return the first choice's text
    return completions#.choices[0].message

def create_messages(questions, instructions):
    """Creates one chat completion template, with the first message being the instructions."""
    messages = []
    sysmessage = instructions
    if sysmessage:
        messages.append(sysmessage)
    if isinstance(questions, str):
        questions = list([questions])
    for q in questions:
        messages.append({"role":"user", "content": q })
    return messages

def prepare_batch(batch, tokenizer):
    '''prepares data by organizing dataloader output into readable observations. input batch should be one iteration of iter(dataloader)'''
    batch_t = np.transpose(np.array([[*tup] for tup in batch]))
    data = pd.DataFrame(batch_t, columns=['prompt','label','question'])
    tokens = [tokenizer.encode(row) for row in data['prompt']]
    #make each row the same number of tokens, torch doesn't support ragged rows
    max_len=max([len(i) for i in tokens])
    prompt_tokens = np.array([ np.pad(tokens[i],
               (0,max_len-len(tokens[i])),
               'constant',
                constant_values=0) for i in range(len(tokens))])

    data['token'] = tokens    
    return data

def attack_batch(model, tokenizer, data, max_len=20):
    '''returns one perturbed question for each input question. 
    Output dict has the same format as get_batch, although labels are irrelevant.'''
    rand = np.random.default_rng()
    attacks = data
    perturbed_questions = list()
    losses = list()
    for i, row in data.iterrows():
        question = row['question']
        prompt = 'Q: ' + question + ' rephrased: '
        cur_ids = torch.LongTensor(tokenizer(prompt)['input_ids']).to(device)
        q_done = False
        for i in range(max_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            losses.append(loss)
            #softmax_logits = torch.exp(logits[-1])/torch.sum(torch.exp(logits[-1])).clone().detach().cpu()#softmax the token probabilities given the last embedding
            #softmax_logits[torch.argmax(softmax_logits)] += 1 - sum(softmax_logits)
            #print(i, sum(softmax_logits.clone().detach().cpu()))
            next_token_id = rand.choice(torch.topk(logits[-1], 3)[1].clone().detach().cpu())#, p=softmax_logits.clone().detach().cpu())
            cur_ids = torch.cat([cur_ids, torch.tensor([next_token_id]).to(device)]) # Add the last word to the running sequence

            if next_token_id in tokenizer('<|endoftext|>') or i == max_len-1:
                q_done = True
                break
        
        if q_done:
            #split attacked question off from sequence, form new prompt
            decoded_q = tokenizer.decode(cur_ids.squeeze(0)).split(': ')[-1] #the generated question
            answers = row['prompt'].split('? ')[1:]
            separator = ' '
            perturbed_prompt = separator.join(np.concatenate((decoded_q, answers), axis=None))
            perturbed_questions.append(perturbed_prompt)
            
    data['prompt'] = perturbed_questions
    losses = torch.stack(losses)
    return data, losses

class RLLoss(torch.nn.Module):
    def __init__(self):
        super(RLLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = (inputs in targets) - 1
        return loss.mean()
    
def tokenize_function(tokenizer, examples):
    return tokenizer(examples['context'])

def log_generations(input_tokens, output_tokens, tokenizer):
    data = {
        'input_text': [],
        'input_tokens': [],
        'output_text': [],
        'output_tokens': [],
        'empty': []
    }

    for input_ids, output_ids in zip(input_tokens, output_tokens):
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        data['input_text'].append(input_text)
        data['input_tokens'].append(input_ids)
        data['output_text'].append(output_text)
        data['output_tokens'].append(output_ids)
        data['empty'].append(output_text == "")

    df = pd.DataFrame(data)
    df.to_csv('output_data.csv', index=False)
    print("Data saved to output_data.csv")

    try:
        # Exploratory Data Analysis
        empty_df = df[df['empty'] == True]
        non_empty_df = df[df['empty'] == False]

        last_tokens = [tokens[-1] for tokens in empty_df['input_tokens']]
        second_last_tokens = [tokens[-2] for tokens in empty_df['input_tokens'] if len(tokens) > 1]
        third_last_tokens = [tokens[-3] for tokens in empty_df['input_tokens'] if len(tokens) > 2]
        second_tokens_empty = [tokens[1] for tokens in empty_df['output_tokens'] if len(tokens) > 1]
        second_tokens_non_empty = [tokens[1] for tokens in non_empty_df['output_tokens'] if len(tokens) > 1]
        first_tokens_non_empty = [tokens[0] for tokens in non_empty_df['output_tokens']]
        first_tokens_empty = [tokens[0] for tokens in empty_df['output_tokens']]

        most_common_last_token = Counter(last_tokens).most_common(1)
        most_common_second_last_token = Counter(second_last_tokens).most_common(1)
        most_common_third_last_token = Counter(third_last_tokens).most_common(1)
        most_common_first_token_empty = Counter(first_tokens_empty).most_common(1)
        most_common_second_token_empty = Counter(second_tokens_empty).most_common(1)
        most_common_first_token_non_empty = Counter(first_tokens_non_empty).most_common(1)
        most_common_second_token_non_empty = Counter(second_tokens_non_empty).most_common(1)

        with open('eda_log.txt', 'w') as log_file:
            log_file.write(f"Most common LAST TOKEN in inputs if empty==True: {most_common_last_token}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_last_token[0][0]])}\n")
            log_file.write(f"Most common 2ND LAST TOKEN in inputs if empty==True: {most_common_second_last_token}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_second_last_token[0][0]])}\n")
            log_file.write(f"Most common 3RD LAST TOKEN in inputs if empty==True: {most_common_third_last_token}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_third_last_token[0][0]])}\n")
            log_file.write(f"Most common FIRST TOKEN in outputs if empty==True: {most_common_first_token_empty}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_first_token_empty[0][0]])}\n")
            log_file.write(f"Most common SECOND TOKEN in outputs if empty==True: {most_common_second_token_empty}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_second_token_empty[0][0]])}\n")
            log_file.write(f"Most common FIRST TOKEN in outputs if empty==False: {most_common_first_token_non_empty}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_first_token_non_empty[0][0]])}\n")
            log_file.write(f"Most common SECOND TOKEN in outputs if empty==False: {most_common_second_token_non_empty}\n")
            log_file.write(f"Decoded: {tokenizer.decode([most_common_second_token_non_empty[0][0]])}\n")

        print("Exploratory data analysis saved to eda_log.txt")
    except:
        print("Error in exploratory data logging")

def llama_chat_template(questions, system_instructions, redteam_instructions, eos_token, prompt_injection):
    bsys_token = '<|start_header_id|>system<|end_header_id|>\n'
    bmsg_token = '<|start_header_id|>user<|end_header_id|>\n'
    header_string = '<|start_header_id|>assistant<|end_header_id|>\n'
    chats = []
    for question in questions:
        chat = bsys_token + system_instructions['content'] + eos_token + \
            bmsg_token + redteam_instructions['content'] + question + eos_token + header_string + prompt_injection
        assert isinstance(chat, str), "chat must be a string"
        chats.append(chat)
    return chats

def mistral_chat_template(questions, system_instructions, redteam_instructions, eos_token, prompt_injection):
    # Mistral chat template format for batch processing
    chats = []
    for question in questions:
        chat = f"<s>[INST] {system_instructions['content']}\n\n{redteam_instructions['content']}{question} [/INST] " + prompt_injection
        assert isinstance(chat, str), "chat must be a string"
        chats.append(chat)
    return chats

def add_mistral_special_tokens(tokenizer):
    # Add special tokens for Mistral
    special_tokens_dict = {
        'pad_token': '<pad>',
        'additional_special_tokens': ['[INST]', '[/INST]', ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def add_llama_special_tokens(tokenizer):
    # Add special tokens for Llama
    pad_token = '<|finetune_right_pad_id|>'
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = pad_token_id
    return tokenizer

def add_deepseek_special_tokens(tokenizer):
    pass


def test_prompt_with_guard(questions, rephrases, save_path='prompts_guard.png'):
    TOKEN = "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"
    model_id = 'meta-llama/Meta-Llama-Guard-2-8B'
    device = 'cuda' 
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    config = AutoConfig.from_pretrained(model_id, token=TOKEN)
    config.pad_token_id = config.eos_token_id
    config.return_dict_in_generate = True
    config.output_logits = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN, config=config)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, token=TOKEN).to(device=device, dtype=dtype)
    original_scores = []
    for q in questions:
        chat = [{'role': 'user', 'content': q}]
        inputs = tokenizer.apply_chat_template(chat, return_tensors='pt').to(device)
        outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs), output_scores=True)
        prompt_len = inputs.shape[-1]
        logits = torch.softmax(outputs.logits[0], dim=-1)

        #print(min(outputs.logits[0][0]), max(outputs.logits[0][0]), outputs.logits[0][0][19193])
        #print(outputs.logits[0][0][19193], tokenizer.decode(outputs.sequences[0][prompt_len:]))
        #assert False
        safe_logits = logits[0][19193]
        original_scores.extend([safe_logits])
    original_scores = torch.tensor(original_scores).detach().cpu().numpy()

    rephrased_scores = []
    for q in rephrases:
        chat = [{'role': 'user', 'content': q}]
        inputs = tokenizer.apply_chat_template(chat, return_tensors='pt').to(device)
        outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs), output_scores=True)
        prompt_len = inputs.shape[-1]
        logits = torch.softmax(outputs.logits[0], dim=-1)

        rephrased_logits = logits[0][19193]
        rephrased_scores.extend([rephrased_logits])
    rephrased_scores = torch.tensor(rephrased_scores).detach().cpu().numpy()

    print(f"Original safety rating: {original_scores.mean()}")
    print(f"Safety rating when adversarially rephrased: {rephrased_scores.mean()}")
    questions_df = pd.DataFrame({'question': questions, 'original_score': original_scores, 'rephrased_score': rephrased_scores})
    plot_rephrase_scores(questions_df, save_path=save_path)

def test_chat_with_guard(questions, rephrases, responses, save_path='responses_guard.png'):
    TOKEN = "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"
    model_id = 'meta-llama/Meta-Llama-Guard-2-8B'
    device = 'cuda' 
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    config = AutoConfig.from_pretrained(model_id, token=TOKEN)
    config.pad_token_id = config.eos_token_id
    config.return_dict_in_generate = True
    config.output_logits = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN, config=config)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, token=TOKEN).to(device=device, dtype=dtype)
    original_scores = []
    for q,a in zip(questions, responses):
        chat = [{'role': 'user', 'content': q},
                {'role': 'assistant', 'content': a}]
        inputs = tokenizer.apply_chat_template(chat, return_tensors='pt').to(device)
        outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs), output_scores=True)
        prompt_len = inputs.shape[-1]
        logits = torch.softmax(outputs.logits[0], dim=-1)

        #print(min(outputs.logits[0][0]), max(outputs.logits[0][0]), outputs.logits[0][0][19193])
        #print(outputs.logits[0][0][19193], tokenizer.decode(outputs.sequences[0][prompt_len:]))
        #assert False
        safe_logits = logits[0][19193]
        original_scores.extend([safe_logits])
    original_scores = torch.tensor(original_scores).detach().cpu().numpy()

    rephrased_scores = []
    for q,a in zip(rephrases, responses):
        chat = [{'role': 'user', 'content': q},
                {'role': 'assistant', 'content': a}]
        inputs = tokenizer.apply_chat_template(chat, return_tensors='pt').to(device)
        outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs), output_scores=True)
        prompt_len = inputs.shape[-1]
        logits = torch.softmax(outputs.logits[0], dim=-1)

        rephrased_logits = logits[0][19193]
        rephrased_scores.extend([rephrased_logits])
    rephrased_scores = torch.tensor(rephrased_scores).detach().cpu().numpy()

    print(f"Original safety rating: {original_scores.mean()}")
    print(f"Safety rating when adversarially rephrased: {rephrased_scores.mean()}")
    questions_df = pd.DataFrame({'question': questions, 'original_score': original_scores, 'rephrased_score': rephrased_scores})
    plot_rephrase_scores(questions_df, save_path=save_path)

def plot_rephrase_scores(questions_df, save_path='safety_scores.png'):
    """Plot histograms of original and rephrased scores"""
    plt.figure(figsize=(10, 6))
    
    # Create histograms
    labels = ['Original Q\'s', 'Rephrased Q\'s']
    plt.hist([questions_df['original_score'], questions_df['rephrased_score']], 
             density=True,
             histtype='bar',
             alpha=0.5, 
             label=labels, 
             bins=20,
             color=['blue', 'red'])
    
    # Add labels and title
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('P(\'safe\') with and without adversarial rephrasing')
    plt.legend()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def print_memory(message=None):
        print('-' * 80)
        if message:
            print(message)
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
        print(f"Max memory cached: {torch.cuda.max_memory_reserved() / 1024 ** 2} MB")
        print('-' * 80)

def build_chat(messages, user_num):
    """user_num = 0 means the user speaks first. The most recent message is the user's message. The red-teaming agent should submit user_num=1, the target should have user_num=0."""
    chat = []
    for i, msg in enumerate(messages):
        if isinstance(msg, list):
            msg = msg[0]
        assert isinstance(msg, str), f"Message must be a string, got {type(msg)}"
        if msg[-4:] == 'user':
            msg = msg[:-4]
        elif msg[-9:] == 'assistant':
            msg = msg[:-9]
        if i % 2 == user_num:
            chat.append({'role': 'user', 'content': msg})
        else:
            chat.append({'role': 'assistant', 'content': msg})
    return chat

def check_device_mem(threshold):
    # Check if allocated memory exceeds max GPU memory
    assert threshold >= 0 and threshold <= 1, "Threshold must be between 0 and 1"
    allocated_memory = torch.cuda.memory_allocated()
    max_memory = torch.cuda.get_device_properties(0).total_memory
    return allocated_memory > max_memory * threshold
        