from markov import DadJokesMarkovChain
from custom_gpt2 import DadJokesGPT

class DadJokesPipeline:
    def __init__(self, **kwargs):
        mc_fullpath: str = kwargs.get('mc_fullpath', None)  
        model_type: str = kwargs.get('model_type', None)
        model_path: str = kwargs.get('model_path', None)
        tokenizer_path: str = kwargs.get('tokenizer_path', None) 
        self.markov_chain = DadJokesMarkovChain(mc_fullpath=mc_fullpath)
        self.transformer = DadJokesGPT(
            model_type=model_type, 
            model_path=model_path, 
            tokenizer_path=tokenizer_path
        )
    
    def train_from_filename(self, data_path: str, column_with_jokes: str):
        self.markov_chain.train_chain(data_path, column_with_jokes)
        
        self.transformer.define_data(data_path, column_with_jokes)
        self.transformer.train_model()

    def save_pipeline(self, directory: str):
        self.markov_chain.dump_chain(directory, 'chain.pkl')
        self.transformer.save_model(directory, 'gpt2')
        self.transformer.save_tokenizer(directory)

    def generate_joke(self):
        prefix = self.markov_chain.generate_prefix_for_gpt2()
        joke = self.transformer.generate(prefix)

        return joke

