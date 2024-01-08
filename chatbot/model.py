import pandas as pd
from transformers import DataCollatorForSeq2Seq, BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from datasets import Dataset, DatasetDict

# Load your CSV data into a DataFrame (replace 'your_file.csv' with your file path)
data = pd.read_csv('./chatbot/dad_jokes_dataset.csv')

# Initialize the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def create_input_sequence(data):
    source_texts = data['Request'].tolist()
    target_texts = data['Joke'].tolist()
    model_inputs = tokenizer(source_texts, padding='max_length', max_length=64, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, padding='max_length', max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = Dataset.from_dict(create_input_sequence(data))

training_args = Seq2SeqTrainingArguments(
    output_dir='./chatbot',  # output directory
    evaluation_strategy="steps",  # set evaluation strategy
    save_strategy="steps",  # set save strategy
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    logging_dir='./chatbot',  # directory for storing logs
    logging_steps=1000,
    eval_steps=1000,
    save_steps=5000,
    warmup_steps=2000,
    weight_decay=0.01,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model='rouge2',
)

# Initialize Trainer for sequence-to-sequence task
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Start training
trainer.train()
