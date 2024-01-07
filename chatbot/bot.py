from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, Updater, CommandHandler, MessageHandler, CallbackContext, CallbackQueryHandler, filters

import random
from random import randint
import datetime
import requests
import configparser
from openai import OpenAI

OPENAI_TOKEN = 'YOUR_TOKEN'
client = OpenAI(api_key=OPENAI_TOKEN) 

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

async def handle_message(update: Update, context: CallbackContext):
    message_type: str = update.message.chat.type
    message: str = update.message.text
    
    print(f'User ({update.message.chat.id}) triggered a handler for {message_type}: {message}')

    options = ['Chat GPT', 'Custom BART']
    # selected = random.choice(options)
    selected = options[0]
    if message_type == 'group':
        if BOT_USERNAME in message:
            new_message: str = message.replace(BOT_USERNAME, '')
            response = make_inference(new_message, selected)   
        else:
            return
    else:
        response = make_inference(message, selected)  
    print(f'Bot: {response}')

    keyboard = [
        [InlineKeyboardButton("{}".format(options[0]), callback_data=str(options[0]))],
        [InlineKeyboardButton("{}".format(options[1]), callback_data=str(options[1]))]
    ] if bool(random.getrandbits(1)) else [
        [InlineKeyboardButton("{}".format(options[1]), callback_data=str(options[1]))],
        [InlineKeyboardButton("{}".format(options[0]), callback_data=str(options[0]))]
    ]
    context.user_data['correct_answer'] = selected
    await update.message.reply_text(response, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='html')  

def make_inference(text, selected_option):

    if selected_option == 'Chat GPT':
        messages = [
            {"role": "system", "content": "You are a comedian that receives request from audience and should generate a joke for it."},
            {"role": "user", "content": text}
        ]
    
        chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        return reply
    else:
        return ''

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
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(CallbackQueryHandler(button))

    app.run_polling(poll_interval=3)
    # updater.idle()


if __name__ == '__main__':
    print("Bot started")
    main()