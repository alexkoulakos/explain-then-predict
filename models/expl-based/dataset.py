import pandas as pd

from torch.utils.data import Dataset

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

        if 'explanation_2' in self.df.keys():
            explanation_2 = self.df['explanation_2'][index].lower()
            explanation_3 = self.df['explanation_3'][index].lower()

            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'explanation_2': explanation_2,
                'explanation_3': explanation_3,
                'label': self.df['label'][index]
            }
        else:
            return {
                'premise': premise,
                'hypothesis': hypothesis,
                'explanation_1': explanation_1,
                'label': self.df['label'][index]
            }