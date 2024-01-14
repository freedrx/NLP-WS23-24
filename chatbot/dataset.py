from torch.utils.data import Dataset

from transformers import GPT2TokenizerFast

import pandas as pd

# Pytorch dataset for given dad jokes dataset
class DadJokesDataset(Dataset):
    def __init__(self, path:str, column_to_read: str, tokenizer: GPT2TokenizerFast):
        data = pd.read_csv(path)
        if column_to_read not in data.columns:
            raise ValueError('False column name provided.')
        data = data[column_to_read].to_list()
        X = [f"<startofstring>{joke}<endofstring>" for joke in data]
        self.num_samples = len(X)
        encoded = tokenizer(X, max_length=64, truncation=True, padding=True, return_tensors="pt")

        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])