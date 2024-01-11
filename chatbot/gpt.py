import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim

from torch.utils.data import Dataset
import pandas as pd


class DadJokesDataset(Dataset):
    def __init__(self, path:str, tokenizer):
        data = pd.read_excel(path)['joke'].to_list()
        self.X = [f"<startofstring>{joke}<endofstring>" for joke in data]
        encoded = tokenizer(self.X, max_length=64, truncation=True, padding=True, return_tensors="pt")
        self.input_ids = encoded['input_ids'][:, :-1]
        self.attention_mask = encoded['attention_mask'][:, 1:]
        self.labels = encoded['input_ids'][:, 1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return ((self.input_ids[idx], self.attention_mask[idx]), self.labels[idx])

def train_model(model, dataloader):
    # model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 12

    for i in tqdm.tqdm(range(epochs)):
        model.train()
        for (joke, attention_mask), labels in dataloader:
            joke = joke.to(device)
            attention_mask = attention_mask.to(device)

            labels = labels.to(device)
            pad_token_id = tokenizer.pad_token_id
            labels[labels == pad_token_id] = -100
            optimizer.zero_grad()
            loss = model(joke, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
        model.eval()
        generated_sentence = inference(model, tokenizer)
        print(f'epoch {i+1}, inference: {generated_sentence}\n')
    torch.save(model.state_dict(), "model_state.pt")

def inference(model, tokenizer):
    text = "<startofstring>"
    tokenized = tokenizer(text, return_tensors='pt')
    out = model.generate(tokenized['input_ids'], attention_mask=tokenized['attention_mask'], max_length=24, do_sample=True, top_k = 50,top_p = 0.9, num_return_sequences = 1)
    decoded = tokenizer.decode(out[0])
    return decoded

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"bos_token": "<startofstring>", "eos_token": "<endofstring>", "pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))
dataloader = DataLoader(DadJokesDataset('./chatbot/cleaned_dataset.xls', tokenizer), batch_size=8)
train_model(model, dataloader)