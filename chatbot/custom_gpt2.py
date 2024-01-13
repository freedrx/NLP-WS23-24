import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2LMHeadModel

import tqdm

from dataset import DadJokesDataset

from datetime import datetime


class DadJokesGPT:
    def __init__(self, model_type: str = None, model_path: str = None, tokenizer_path: str = None):
        if (model_type is None) and (model_path is None) and (tokenizer_path is None):
            raise ValueError(
                'You should either provide a path to existing model' \
                ' or load the pretrained one using model type.'
            )
        if (model_type is not None) and (model_path is not None) and (tokenizer_path is not None):
            raise ValueError(
                'It is not possible to load both pretrained and local model simultaniously.'
            )
        if any([component is None for component in [model_path, tokenizer_path]]):
            raise ValueError(
                'Both tokenizer path and model path should be specified.'
            )
        self.dataloader = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type:
            self._init_using_pretrained(model_type=model_type)
        else:
            self._init_using_local_components(
                model_path=model_path,
                tokenizer_path=tokenizer_path
            )

    def _init_using_pretrained(self, model_type: str):
        if model_type not in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            raise ValueError('Inappropriate gpt2 type provided.')
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_type)
        self.tokenizer.add_special_tokens({
            "bos_token": "<startofstring>", 
            "eos_token": "<endofstring>", 
            "pad_token": "[PAD]"
        })

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_type).to(self.device)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

    def _init_using_local_components(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

    def generate(self, text: str = '', max_length_of_generated_text: int = 32, num_return_sequences: int = 1):
        if any(value <= 0 for value in [max_length_of_generated_text, num_return_sequences]):
            raise ValueError('Maximal length and number of return sequences must be a positive integer.')
        
        prefix = "<startofstring>" + text.strip()
        tokenized_prefix = self.tokenizer(prefix, return_tensors='pt')
        out = self.gpt2.generate(
            tokenized_prefix['input_ids'].to(self.device), 
            attention_mask=tokenized_prefix['attention_mask'].to(self.device), 
            max_length=max_length_of_generated_text,
            do_sample=True, 
            top_k = 50,
            top_p = 0.9, 
            num_return_sequences = num_return_sequences)
        
        generated_sequences = [
            self.tokenizer.decode(encoded_sequence)
            .replace('<startofstring>', '')
            .replace('<endofstring>', '') 
            for encoded_sequence in out
        ]

        return generated_sequences
    
    def train_model(self, directory_for_provisional_models: str = None):
        formated_current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        optimizer = optim.Adam(self.gpt2.parameters(), lr=0.001)
        epochs = 20
        train_losses = []
        for i in tqdm.tqdm(range(epochs)):
            train_loss = 0.0
            self.gpt2.train()
            for joke, attention_mask in self.dataloader:
                joke = joke.to(self.device)
                attention_mask = attention_mask.to(self.device)

                labels = joke.clone().to(self.device)
                pad_token_id = self.tokenizer.pad_token_id
                labels[labels == pad_token_id] = -100

                optimizer.zero_grad()
                loss = self.gpt2(joke, attention_mask=attention_mask, labels=labels).loss    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(self.dataloader)
            train_losses.append(train_loss)
            print(f'\n{train_loss}\n')
            self.gpt2.eval()
            generated_sentence = self.generate()
            print(f'\nepoch {i+1}, inference: {generated_sentence}\n')
            if directory_for_provisional_models:
                self.save_model(
                    directory=directory_for_provisional_models,
                    filename= 'custom_gpt' + formated_current_time
                )
        self.flush_data()
    
    def define_data(self, data_path: str, column_to_read: str, batch_size: int = 32):
        if data_path[-4:] != '.csv':
            raise ValueError('Expected csv format.')
        self.dataloader = DataLoader(DadJokesDataset(data_path, column_to_read, self.tokenizer), batch_size=batch_size)
    
    def save_model(self, directory: str, filename: str):
        torch.save(self.gpt2.state_dict(), f"{directory}/{filename}.pt")
    
    def save_tokenizer(self, directory: str):
        self.tokenizer.save_pretrained(f'{directory}/tokenizer_gpt2')
    
    def flush_data(self):
        self.dataloader = None
