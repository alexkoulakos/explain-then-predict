import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from textattack.models.wrappers import ModelWrapper

class BertNLIModel(nn.Module):
    def __init__(self, checkpoint, device):
        super(BertNLIModel, self).__init__()

        self.device = device

        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.embedding_dim = self.encoder.config.to_dict()['hidden_size']
        self.output_dim = 3
        
        self.classifier = nn.Linear(self.embedding_dim, self.output_dim)
    
    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        _, embeddings = self.encoder(
            input_ids,
            attention_mask,
            return_dict=False
        )

        label_distributions = self.classifier(embeddings)
        
        return label_distributions

class BertNLITokenizer():
    def __init__(self, checkpoint, max_length):
        self.checkpoint = checkpoint
        self.max_length = max_length 

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)
    
    def encode(self, premise, hypothesis):
        return self.tokenizer(
            premise,
            hypothesis,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
    
    def __call__(self, premise, hypothesis):
        return self.encode(premise, hypothesis)

class BertNLIModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        super(BertNLIModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
    
    # inputs: [(premise_1, hypothesis_1), (premise_2, hypothesis_2), ...]
    def __call__(self, inputs):
        # print(inputs)

        premises, hypotheses = [], []

        for i in range(len(inputs)):
            if not isinstance(inputs[i], tuple):
                print(i)
                premise, hypothesis = inputs[i].split('<SPLT>')
            else:
                premise = inputs[i][0]
                hypothesis = inputs[i][1]
            
            premises.append(premise)
            hypotheses.append(hypothesis)

        encoder_input = self.tokenizer(premises, hypotheses)
        
        input_ids = encoder_input['input_ids']
        attention_mask = encoder_input['attention_mask']

        with torch.no_grad():
            return self.model(input_ids, attention_mask)
            
class EsnliDataset(Dataset):
    def __init__(self, data_path, rows=-1):
        if rows == -1:
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv(data_path)[:rows]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        premise = self.df['premise'][index].lower()
        hypothesis = self.df['hypothesis'][index].lower()
        explanation_1 = self.df['explanation_1'][index].lower()
        label = self.df['label'][index]

        if 'explanation_2' in self.df.keys():
            explanation_2 = self.df['explanation_2'][index].lower()
            explanation_3 = self.df['explanation_3'][index].lower()

            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'explanation_2': explanation_2,
                'explanation_3': explanation_3,
                'label': label
            }
        else:
            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'label': label
            }