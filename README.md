Group members: 
- Yernar Kubegenov
- Alexander Prisak
- Kilian Hümmer
- Mykhailo Druhov

Our group members have made an even contribution for this portfolio.

This repository contains portfolio for the course Semantic Data Processing and Representation
and contains following items:
 - Presentation of GloVe
 - Presentation of ALBERT
 - Markovify example
 - Movie Classifier
 - Project (Telegram ChatBot)


Description of the final project:

Our group decided to implement Telegram Bot. The purpose of this bot is generation of dad jokes. The generated joke can be created by one of three models - ChatGPT, custom Huggingface GPT2 + Markov Chain or GPT2 from gpt-2-simple package. User has to guess which model has generated the given joke. 

Setup:
1. You have to download our finetuned Huggingface GPT2 and GPT2-Simple from THWS Cloud. Link: https://cloud.thws.de/s/4sQyPb7ERs265Qg
2. After unzipping models You have to alter path to the models in bot.py file. ./custom folder contains Huggingface GPT2. checkpoint - GPT2 Simple (don't change run_name param in file).
3. Execute the code of bot.py locally to start server side of the bot
4. Find our bot in Telegram using its nickname @thws23maibot
5. Enjoy