from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, Updater, CommandHandler, MessageHandler, CallbackContext, CallbackQueryHandler, filters

import random
import datetime

from openai import OpenAI

from pipeline import DadJokesPipeline
from simple_gpt2 import GPT2Simple


OPENAI_TOKEN = 'YOUR_TOKEN'
client = OpenAI(api_key=OPENAI_TOKEN) 

custom_gpt2_pipeline = DadJokesPipeline(
    model_path='./chatbot/custom/gpt2_model',
    tokenizer_path='./chatbot/custom/tokenizer_gpt2',
    mc_fullpath='./chatbot/custom/chain.pkl'
)

simple_gpt2 = GPT2Simple(
    checkpoint_dir='./chatbot/checkpoint',
    run_name='dadjokes'
)

# config = configparser.ConfigParser()
# config.read('config.ini')

# BOT_TOKEN = config.get('default', 'BOT_TOKEN')
# weather_key = config.get('default', 'weather_key')
BOT_TOKEN: Final = 'YOUR TOKEN'
BOT_USERNAME: Final = '@thws23maibot'
 
async def start(update: Update, context: CallbackContext) -> None:
    text = "Hello! Give me a dad joke request and I generate one for you!"
    await update.message.reply_text(text, parse_mode="HTML")

async def time(update: Update, context: CallbackContext) -> None:
    text = "Received! Day and time: " + str(datetime.datetime.now())
    await update.message.reply_text(text, parse_mode="HTML")

async def help(update: Update, context: CallbackContext) -> None:
    text = 'Enter a message that contains request for dad joke and I do the rest!\nYour joke should be provided in a format "Tell/make a joke about {topic name}"' 
    await update.message.reply_text(text, parse_mode="HTML")

async def joke(update: Update, context: CallbackContext):
    message_type: str = update.message.chat.type
    message: str = update.message.text
    
    print(f'User ({update.message.chat.id}) triggered a handler for {message_type}: {message}')

    options = ['Chat GPT', 'GPT2 Custom', 'GPT2 Simple']
    random_option = random.choice(options)

    response = make_inference(random_option)  
    print(f'Bot: {response}')

    keyboard = [[InlineKeyboardButton("{}".format(option), callback_data=str(options))] for option in options]
    random.shuffle(keyboard)

    context.user_data['correct_answer'] = random_option
    await update.message.reply_text(response, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='html')  

def make_inference(selected_option):

    if selected_option == 'Chat GPT':
        messages = [
            {"role": "system", "content": "You are a comedian that generates bad jokes."},
            {"role": "user", "content": "Provide a dadjoke"}
        ]
    
        chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        return reply
    elif selected_option == 'GPT2 Custom':
        jokes_list = custom_gpt2_pipeline.generate_joke()
        return jokes_list[0]
    else:
        return simple_gpt2.generate()

async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    choice = query.data
    correct_answer = context.user_data['correct_answer']
    print(choice)

    if choice == correct_answer:
        await query.edit_message_text("You've guessed correctly", parse_mode='html')
    else:
        await query.edit_message_text(f"Your choice is incorrect. The joke was created by {correct_answer}", parse_mode='html')


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help))
    app.add_handler(CommandHandler('time', time))
    app.add_handler(CommandHandler('joke', joke))
    app.add_handler(CallbackQueryHandler(button))

    app.run_polling(poll_interval=3)
    # updater.idle()


if __name__ == '__main__':
    print("Bot started")
    main()