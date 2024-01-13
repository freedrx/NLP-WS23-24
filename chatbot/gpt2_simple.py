
import gpt_2_simple as gpt2
gpt2.download_gpt2(model_name="124M")
file_name = "./chatbot/cleaned_dataset.csv"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=500,
              restore_from='fresh',
              run_name='dadjokes',
              print_every=10,
              sample_every=100,
              )

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='dadjokes')
print(gpt2.generate(
    sess,
    run_name='dadjokes',
    length=64,
    temperature=0.6,
    nsamples=5,
    batch_size=5
))