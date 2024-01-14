import gpt_2_simple as gpt2


class GPT2Simple:
    def __init__(self, checkpoint_dir: str, run_name: str):
        self.session = gpt2.start_tf_sess()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        gpt2.load_gpt2(self.session, checkpoint_dir=self.checkpoint_dir, run_name=self.run_name)

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

        return self.extract_substring(out)

    @classmethod
    def extract_substring(cls, text):
        start_marker = "<|startoftext|>"
        end_marker = "<|endoftext|>"
        start_index = text.find(start_marker)
        end_index = text.find(end_marker, start_index + len(start_marker))

        if start_index != -1 and end_index != -1:
            return text[start_index + len(start_marker):end_index]
        else:
            return text

