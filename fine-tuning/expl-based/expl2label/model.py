import torch.nn as nn

from transformers import AutoModel

class Expl2LabelModel(nn.Module):
    def __init__(self, checkpt):
        super(Expl2LabelModel, self).__init__()

        self.encoder = AutoModel.from_pretrained(checkpt)
        self.embedding_dim = self.encoder.config.to_dict()['hidden_size']
        self.output_dim = 3
        
        self.classifier = nn.Linear(self.embedding_dim, self.output_dim)
    
    def forward(self, input_ids, attention_mask):
        _, embeddings = self.encoder(
            input_ids,
            attention_mask,
            return_dict=False
        )

        return self.classifier(embeddings)