import re
import pandas as pd
import time
import emoji

"""
Datacollection Process:

We gathered the dad jokes data from kaggle dataset: https://www.kaggle.com/datasets/oktayozturk010/reddit-dad-jokes

We used this code to clean the data and remove any jokes that were not suitable for our model. We i.e. removed any emojis and contractions from the dataset or removed jokes that were too long or too short.

"""


class DadJokesCleaner:
    def __init__(self, csv_file_path):        
        self.csv_file_path = csv_file_path        

    def remove_emoji(self, string):
        return emoji.demojize(string)

    def clean_data(self):
        df = pd.read_csv(self.csv_file_path)
        df = df[['joke']]

        df['joke'] = df['joke'].str.replace('\n', '') # remove new lines
        df['joke'] = df['joke'].str.replace('\r', '') # remove new lines
        df['joke'] = df['joke'].str.replace('\t', '') # remove tabs
        df['joke'] = df['joke'].str.replace('\xa0', '') # remove non-breaking space
        df['joke'] = df['joke'].str.replace('\"', '') # remove double quotes
        df['joke'] = df['joke'].str.replace('\'', '') # remove single quotes

        df = df.dropna()
        df = df.drop_duplicates()
        df = df[df['joke'].str.len() > 10] # remove jokes that are too short
        df = df[df['joke'].str.len() < 500] # remove jokes that are too long
        df = df[df['joke'].str.count('\.') < 3] # remove jokes with too many dots
        df = df[df['joke'].str.count('\n') < 3] # remove jokes with too many new lines
        df = df[df['joke'].str.count('\?') < 3] # remove jokes with too many question marks
        df = df[df['joke'].str.count('\!') < 3] # remove jokes with too many exclamation marks
        df = df[df['joke'].str.count('\,') < 3] # remove jokes with too many commas
        df = df[df['joke'].str.count('\:') < 3] # remove jokes with too many colons
        df = df[df['joke'].str.count('\;') < 3] # remove jokes with too many semicolons
        df = df[df['joke'].str.count('\-') < 3] # remove jokes with too many dashes
        df = df[df['joke'].str.count('\(') < 3] # remove jokes with too many opening brackets
        df = df[df['joke'].str.count('\[') < 3] # remove jokes with too many opening brackets
        df = df[df['joke'].str.count('\{') < 3] # remove jokes with too many opening brackets
        df = df[df['joke'].str.count('\#') < 3] # remove jokes with too many hashtags
        df = df[df['joke'].str.count('\*') < 3] # remove jokes with too many stars
        df = df[df['joke'].str.count('\_') < 3] # remove jokes with too many underscores
        df = df[df['joke'].str.count('\/') < 3] # remove jokes with too many slashes
        df = df[df['joke'].str.count('\\\\') < 3] # remove jokes with too many backslashes
        df = df[df['joke'].str.count('\$') < 3] # remove jokes with too many dollar signs
        df = df[df['joke'].str.count('\%') < 3] # remove jokes with too many percent signs
        df = df[df['joke'].str.count('\&') < 3] # remove jokes with too many ampersands
        df = df[df['joke'].str.count('\@') < 3] # remove jokes with too many at signs
        df = df[df['joke'].str.count('\+') < 3] # remove jokes with too many plus signs
        df = df[df['joke'].str.count('\=') < 3] # remove jokes with too many equal signs
        df = df[df['joke'].str.count('\>') < 3] # remove jokes with too many greater than signs
        df = df[df['joke'].str.count('\>') < 3] # remove jokes with too many less than signs
        df = df[df['joke'].str.count('\~') < 3] # remove jokes with too many tildes

        df['joke'] = df['joke'].apply(lambda x: self.remove_emoji(x)) # remove emojis

        # remove contractions
        contractions_dict = {
            "ain't": "are not", "'s": " is", "aren't": "are not",
            "can't": "cannot", "can't've": "cannot have",
            "'cause": "because", "could've": "could have", "couldn't": "could not",
            "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will", "it'll've": "it will have",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "might've": "might have", "mightn't": "might not", "must've": "must have",
            "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not",
            "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would",
            "she'll": "she will", "she'll've": "she will have", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
            "that'd": "that would", "that's": "that is", "there'd": "there had",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they'll've": "they will have", "they're": "they are", "they've": "they have",
        }
        df['joke'] = df['joke'].replace(contractions_dict, regex=True) # replace contractions

        df.to_csv('cleaned_dataset1.csv', index=False) # save cleaned dataset


#Execute the code
csv_file_path = r'C:\Users\49174\Documents\Studium - AI\Semester II\NLP - Semantic Data\Group project\fine-tuning\reddit_dadjokes.csv'
djc = DadJokesCleaner(csv_file_path)
djc.clean_data()
