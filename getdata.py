import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .helpers import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
from openai import OpenAI
client = OpenAI(organization='org-J57WTuqjaC5PSIhlQhmfgmd9', api_key='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',)

    
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, AutoTokenizer, GPT2Model
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

class MCQADataset(Dataset): #from MCQA github

    def __init__(self,
                   csv_path,
                   use_context=True):
    #     self.dataset = dataset['train'] if training == True else dataset['test']
            self.dataset = pd.read_csv(csv_path)
            self.use_context = use_context

    def __len__(self):
            return len(self.dataset)

    def __getitem__(self,idx):
        return_tuple = tuple()
        if self.use_context:
            context = self.dataset.loc[idx,'exp']
            return_tuple+=(context,)
        question = self.dataset.loc[idx,'question']
        options = self.dataset.loc[idx,['opa', 'opb', 'opc', 'opd']].values
        label = self.dataset.loc[idx,'cop'] - 1
        return_tuple+=(question,options,label)
        return return_tuple
    
class TQADataset(Dataset): #from MCQA github

    def __init__(self,
                   csv_path,
                   use_context=True):
    #     self.dataset = dataset['train'] if training == True else dataset['test']
            self.dataset = pd.read_csv(csv_path)

    def __len__(self):
            return len(self.dataset)

    def __getitem__(self,idx):
        rand = np.random.default_rng()
        question = self.dataset.loc[idx,'Question']
        options = self.dataset.loc[idx,['Correct Answers', 'Incorrect Answers']].values
        separator = '; '
        options = np.concatenate([x.split(separator) for x in options])
        rand.shuffle(options)

        label = self.dataset.loc[idx, 'Best Answer']#np.where(options == self.dataset.loc[idx,'Best Answer'])[0][0]
        answers = separator.join([x for x in options]) 
        if answers[-1] != ';':
            answers += ';'
        prompt = question + ' Select only one of the following answers: ' + answers
        return_tuple = (prompt,label,question)
        return return_tuple

