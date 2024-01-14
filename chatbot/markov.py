import markovify
import pickle

import random

import pandas as pd

class DadJokesMarkovChain:
    def __init__(self, mc_fullpath: str = None):
        if mc_fullpath:
            self.load_chain(filename_with_path=mc_fullpath)
        else:
            self.markov_chain = None
    
    def train_chain(self, path: str, column_to_read: str):
        if path[-4:] != '.csv':
            raise ValueError('Expected csv format.')
        df = pd.read_csv(path)
        
        if column_to_read not in df.columns:
            raise ValueError('False column name provided.')
        text_corpus = ' '.join(df[column_to_read].dropna().astype(str))

        self.markov_chain = markovify.Text(text_corpus)
    
    def generate_prefix_for_gpt2(self):
        if self.markov_chain is None:
            raise ValueError('Before generation you should either load or train a Markov chain.')
        possible_prefix_length = [3, 4, 5]
        generated_words = None

        while(True):
            out = self.markov_chain.make_sentence()
            if out is not None:
                generated_words = out.split()
                break
        
        selected_prefix_length = random.choice(possible_prefix_length)
        retrieved_first_words = generated_words[:selected_prefix_length] if len(generated_words) >= selected_prefix_length else generated_words
        
        return ' '.join(retrieved_first_words) 

    def dump_chain(self, directory: str, filename: str):
        if self.markov_chain is None:
            raise ValueError('Currently you don\'t have any active Markov chain.')
        self._manipulate_chain(
            filename_with_path=directory+filename,
            markov_chain=self.markov_chain,
            action_type='wb'
        )
    
    def load_chain(self, filename_with_path: str):
        self.markov_chain = self._manipulate_chain(
            filename_with_path = filename_with_path,
            action_type='rb'  
        )
    
    @classmethod
    def _manipulate_chain(cls, filename_with_path: str, action_type: str, markov_chain: markovify.text.Text = None):
        if filename_with_path[-4:] != '.pkl':
            raise ValueError('Expected pkl format.')
        if action_type not in ['rb', 'wb']:
            raise ValueError('Incorrect action type is provided')
        if action_type == 'wb' and markov_chain is None:
            raise ValueError('For dumping procedure a Markov chain object is required')
        with open(filename_with_path, action_type) as file:
            if action_type == 'wb':
                pickle.dump(markov_chain, file)
                return
            else:
                chain = pickle.load(file)
                return chain 
