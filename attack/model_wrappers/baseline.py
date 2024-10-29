import torch

from textattack.models.wrappers import ModelWrapper

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
            # print(inputs[i])
            
            if isinstance(inputs[i], tuple):
                premise = inputs[i][0]
                hypothesis = inputs[i][1]
            else:
                if '<SPLT>' in inputs[i]:
                    premise, hypothesis = inputs[i].split('<SPLT>')
                elif '<SPL[UNK]T>' in inputs[i]:
                    premise, hypothesis = inputs[i].split('<SPL[UNK]T>')
            
            premises.append(premise)
            hypotheses.append(hypothesis)

        encoder_input = self.tokenizer(premises, hypotheses)
        
        input_ids = encoder_input['input_ids']
        attention_mask = encoder_input['attention_mask']

        with torch.no_grad():
            return self.model(input_ids, attention_mask)