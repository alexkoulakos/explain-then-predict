from transformers import AutoTokenizer

class Expl2LabelTokenizer():
    def __init__(self, checkpt, max_len):
        self.checkpt = checkpt
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpt, use_fast=True)
    
    def encode(self, explanations):
        return self.tokenizer(
            explanations,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
    
    def __call__(self, explanations):
        return self.encode(explanations)