import torch.nn as nn

from transformers import AutoModel

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