import markovify
import pickle

import random

import pandas as pd

# Class for handling Markov Chain trained on dad jokes
class DadJokesMarkovChain:
    def __init__(self, mc_fullpath: str = None):
        # It is possible to specify existing chain
        if mc_fullpath:
            self.load_chain(filename_with_path=mc_fullpath)
        else:
            self.markov_chain = None
    
    # The method retrieves data from specific column of csv file,
    # compounds them in one corpus text and passes to MC for training
    def train_chain(self, path: str, column_to_read: str):
        if path[-4:] != '.csv':
            raise ValueError('Expected csv format.')
        df = pd.read_csv(path)
        
        if column_to_read not in df.columns:
            raise ValueError('False column name provided.')
        text_corpus = ' '.join(df[column_to_read].dropna().astype(str))

        self.markov_chain = markovify.Text(text_corpus)
    
    # Main objective of Markov chain in DadJokes pipeline is generation of
    # prefix - empty string or a couple words - that will be passed to gpt2 model as 
    # input sequence
    def generate_prefix_for_gpt2(self):
        if self.markov_chain is None:
            raise ValueError('Before generation you should either load or train a Markov chain.')
        
        # define possible length of prefix
        possible_prefix_length = [0, 1, 2]
        generated_words = None

        # while loop to omit None as MC output
        while(True):
            out = self.markov_chain.make_sentence()
            if out is not None:
                # creating bag of words
                generated_words = out.split()
                break
        #random choice of prefix length
        selected_prefix_length = random.choice(possible_prefix_length)
        if selected_prefix_length > 0:
            retrieved_first_words = generated_words[:selected_prefix_length] if len(generated_words) >= selected_prefix_length else generated_words
            return ' '.join(retrieved_first_words) 
        else:
            return ''

    # The method is responsible for saving Markov chain locally when trained
    def dump_chain(self, directory: str, filename: str):
        if self.markov_chain is None:
            raise ValueError('Currently you don\'t have any active Markov chain.')
        self._manipulate_chain(
            filename_with_path=directory+filename,
            markov_chain=self.markov_chain,
            action_type='wb'
        )
    
    # The method is responsible for loading existing Markov chain
    def load_chain(self, filename_with_path: str):
        self.markov_chain = self._manipulate_chain(
            filename_with_path = filename_with_path,
            action_type='rb'  
        )
    
    # The method provides IO-handling for Markov Chain
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
