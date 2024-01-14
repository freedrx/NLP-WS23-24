import gpt_2_simple as gpt2

# The script is responsible for training process of gpt2 simple model

gpt2.download_gpt2(model_name="124M")
file_name = "./chatbot/cleaned_dataset.csv"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
    dataset=file_name,
    model_name='124M',
    steps=500,
    restore_from='fresh',
    model_dir='./chatbot/models',
    checkpoint_dir='./chatbot/checkpoint',
    run_name='dadjokes',
    print_every=10,
    sample_every=100,
)
