import datasets

from torch.utils.data import Dataset

class SNLIDataset(Dataset):
    def __init__(self, split, rows=-1):
        self.snli = datasets.load_dataset(
            "snli", 
            split=split,
            trust_remote_code=True
        ).filter(lambda example: example['label'] in [0, 1, 2])

        if rows != -1:
            self.snli = self.snli[:rows]
    
    def __len__(self):
        return len(self.snli['premise'])
    
    def __getitem__(self, index):
        premise = self.snli['premise'][index].lower()
        hypothesis = self.snli['hypothesis'][index].lower()
        label = self.snli['label'][index]

        return {
            'premise': premise,
            'hypothesis': hypothesis,
            'label': label
        }