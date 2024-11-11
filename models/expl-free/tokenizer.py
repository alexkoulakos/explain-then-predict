from transformers import AutoTokenizer

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