import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_header", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    data = pd.read_csv(args.data_path)
    prompts = data[args.prompt_header]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

