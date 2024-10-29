import torch
import sys

sys.path.append("../")

from textattack.models.wrappers.model_wrapper import ModelWrapper

from models.explain_then_predict import ExplainThenPredictModel

class ExplainThenPredictModelWrapper(ModelWrapper):
    def __init__(self, model: ExplainThenPredictModel, tokenizer):
        super(ExplainThenPredictModelWrapper, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
    
    # inputs: [(premise_1, hypothesis_1), (premise_2, hypothesis_2), ...]
    def __call__(self, inputs):
        premises, hypotheses = [], []

        for i in range(len(inputs)):
            print(inputs[i])
            
            if isinstance(inputs[i], tuple):
                premise = inputs[i][0]
                hypothesis = inputs[i][1]
            else:
                sep_str = ""
                found = False

                for char in inputs[i]:
                    if char == '<':
                        found = True
                    
                    if found:    
                        sep_str += char
                    
                    if char == '>':
                        break
                
                premise, hypothesis = inputs[i].split(sep_str)

                """if '<SPLT>' in inputs[i]:
                    premise, hypothesis = inputs[i].split('<SPLT>')
                elif '<SPL[UNK]T>' in inputs[i]:
                    premise, hypothesis = inputs[i].split('<SPL[UNK]T>')"""
            
            premises.append(premise)
            hypotheses.append(hypothesis)

        encoder_input = self.tokenizer(
            premises, 
            hypotheses,
            padding='max_length',
            max_length=self.model.encoder_max_len,
            truncation=True,
            return_tensors='pt'
        )
            
        input_ids = encoder_input['input_ids']
        attention_mask = encoder_input['attention_mask']

        with torch.no_grad():
            _, pred_labels = self.model(input_ids, attention_mask)

        return pred_labels