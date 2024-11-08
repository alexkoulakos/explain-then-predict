import datasets

from torch.utils.data import Dataset

class EsnliDataset(Dataset):
    def __init__(self, split, rows=-1):
        self.esnli = datasets.load_dataset(
            "esnli", 
            split=split,
            trust_remote_code=True
        ).filter(lambda example: example['label'] in [0, 1, 2])

        if rows != -1:
            self.esnli = self.esnli[:rows]
    
    def __len__(self):
        return len(self.esnli)
    
    def __getitem__(self, index):
        premise = self.esnli['premise'][index].lower()
        hypothesis = self.esnli['hypothesis'][index].lower()
        explanation_1 = self.esnli['explanation_1'][index].lower()

        if 'explanation_2' in self.esnli.keys():
            explanation_2 = self.esnli['explanation_2'][index].lower()
            explanation_3 = self.esnli['explanation_3'][index].lower()

            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'explanation_2': explanation_2,
                'explanation_3': explanation_3,
                'label': self.esnli['label'][index]
            }
        else:
            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'label': self.esnli['label'][index]
            }