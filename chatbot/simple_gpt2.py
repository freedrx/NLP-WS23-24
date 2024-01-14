import gpt_2_simple as gpt2

from utils import extract_substring

# import time

# Class for interacting with GPT-2 model using gpt_2_simple library
class GPT2Simple:
    def __init__(self, checkpoint_dir: str, run_name: str):
        self.session = gpt2.start_tf_sess()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        gpt2.load_gpt2(self.session, checkpoint_dir=self.checkpoint_dir, run_name=self.run_name)

    # Generates jokes using the GPT-2 simple model
    def generate(self, max_length=128, amount_of_jokes=1):
        out = gpt2.generate(
            self.session,
            checkpoint_dir=self.checkpoint_dir,
            run_name=self.run_name,
            length=max_length,
            temperature=0.6,
            nsamples=amount_of_jokes,
            batch_size=1,
            return_as_list=True
        )[0]

        return extract_substring(out, "<|startoftext|>", "<|endoftext|>")

# !!! If you want to test the single component without bot functionality,
# remove comments and start the following code:

# simple_gpt2 = GPT2Simple(
#     checkpoint_dir='./chatbot/checkpoint',
#     run_name='dadjokes'
# )

# while True:
#     print(simple_gpt2.generate())
#     time.sleep(3)