import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd

import tqdm

class DadJokesDataset(Dataset):
    def __init__(self, path:str, tokenizer):
        data = pd.read_csv(path)['joke'].to_list()[:15000]
        X = [f"<startofstring>{joke}<endofstring>" for joke in data]
        self.num_samples = len(X)
        encoded = tokenizer(X, max_length=64, truncation=True, padding=True, return_tensors="pt")

        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

def train_model(model, dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    train_losses = []
    for i in tqdm.tqdm(range(epochs)):
        train_loss = 0.0
        model.train()
        for joke, attention_mask in dataloader:
            joke = joke.to(device)
            attention_mask = attention_mask.to(device)

            labels = joke.clone().to(device)
            pad_token_id = tokenizer.pad_token_id
            labels[labels == pad_token_id] = -100
            optimizer.zero_grad()
            loss = model(joke, attention_mask=attention_mask, labels=labels).loss    
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(dataloader)
        train_losses.append(train_loss)
        print(f'\n{train_loss}\n')
        model.eval()
        generated_sentence = inference(model, tokenizer)
        print(f'epoch {i+1}, inference: {generated_sentence}\n')


def inference(model, tokenizer):
    text = "<startofstring>"
    tokenized = tokenizer(text, return_tensors='pt')
    out = model.generate(tokenized['input_ids'].to(device), attention_mask=tokenized['attention_mask'].to(device), max_length=24, do_sample=True, top_k = 50,top_p = 0.9, num_return_sequences = 1)
    decoded = tokenizer.decode(out[0])
    return decoded

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"bos_token": "<startofstring>", "eos_token": "<endofstring>", "pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))

dataloader = DataLoader(DadJokesDataset('./chatbot/cleaned_dataset.csv', tokenizer), batch_size=8)
train_model(model, dataloader)