from markov import DadJokesMarkovChain
from custom_gpt2 import DadJokesGPT
# import time

# Pipeline class for handling the training and generation process of dadjokes gpt2
class DadJokesPipeline:
    def __init__(self, **kwargs):
        # Extracting predefined Markov chain if specified
        mc_fullpath: str = kwargs.get('mc_fullpath', None) 

        # Retrieve parameters for gpt2 model 
        model_type: str = kwargs.get('model_type', None)
        model_path: str = kwargs.get('model_path', None)
        tokenizer_path: str = kwargs.get('tokenizer_path', None) 
        
        # Initialising the pipeline components
        self.markov_chain = DadJokesMarkovChain(mc_fullpath=mc_fullpath)
        self.transformer = DadJokesGPT(
            model_type=model_type, 
            model_path=model_path, 
            tokenizer_path=tokenizer_path
        )
    
    # The method gets path of csv-file as input and specific column and
    # executes training process of both gpt2 and Markov chain
    def train_from_filename(self, data_path: str, column_with_jokes: str):
        self.markov_chain.train_chain(data_path, column_with_jokes)
        
        self.transformer.define_data(data_path, column_with_jokes)
        self.transformer.train_model()

    # Saving the trained Markov Chain and GPT-2 components to a specified directory
    def save_pipeline(self, directory: str):
        self.markov_chain.dump_chain(directory, 'chain.pkl')
        self.transformer.save_model(directory, 'gpt2')
        self.transformer.save_tokenizer(directory)

    # Generating a joke by combining Markov Chain text and GPT-2 model
    def generate_joke(self):
        prefix = self.markov_chain.generate_prefix_for_gpt2()
        joke = self.transformer.generate(prefix)

        return joke

# !!! If you want to test the single component without bot functionality,
# remove comments and start the following code:
    
# pipeline = DadJokesPipeline(
#     model_path='./chatbot/custom/gpt2_model',
#     tokenizer_path='./chatbot/custom/tokenizer_gpt2',
#     mc_fullpath='./chatbot/custom/chain.pkl'
# )

# while True:
#     print(pipeline.generate_joke())
#     time.sleep(3)
    