import pandas as pd
import emoji



"""
Datacollection Process:

We gathered the dad jokes data from kaggle dataset: https://www.kaggle.com/datasets/oktayozturk010/reddit-dad-jokes

We used this code to clean the data and remove any jokes that were not suitable for our model. We i.e. removed any emojis and contractions from the dataset or removed jokes that were too long or too short.

We filtered the code for any jokes that contained offensive words or phrases.
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


        # remove words that are not suitable for our model
        filter_dict = {
            "sex", "dick", "cock", "penis", "pussy", "nazi", "hitler", 
            "homosexual", "homo", "vagina", "fuck", "shit", "ass", 
            "bastard", "bitch", "cunt", "whore", "slut", "fag", 
            "dyke", "retard", "idiot", "stupid", "moron", "douchebag",
            "n-word", "kike", "spic", "chink", "gook", "wop", "gay", "nigga",
            "cracker", "redneck", "hillbilly", "white trash", 
            "terrorist", "jihad", "isis", "kkk", "murderer", "rapist",
            "abortion", "al-Qaeda", "anal", "anarchy", "anus", "arson", 
            "asshole", "bastard", "bigot", "blasphemy", "bolshevik", 
            "bong", "booze", "cocaine", "commie", "crack", "damn", 
            "date rape", "devil-worship", "dildo", "drugs", "ecstasy", 
            "fascist", "fatso", "felching", "fellatio", "fetish", 
            "fornicate", "gangbang", "genital", "god damn", "goddamn", 
            "guro", "heroin", "homicide", "incest", "jackass", "jerk", "weapon",
            "jigaboo", "kinky", "klan", "lesbian", "masturbate", 
            "meth", "molest", "muff", "murder", "naked", "necrophilia", 
            "nipple", "opiate", "opium", "orgasm", "pedophile", "piss", 
            "poop", "prostitute", "pubic", "queer", "racist", 
            "rape", "rectum", "satanic", "scat", "semen", "sexism", 
            "sexual", "shroom", "snorting", "sodomize", "sodomy", 
            "spastic", "suicide", "terror", "threesome", "toke", 
            "torture", "toxic", "transexual", "underage", "urinate", 
            "urine", "vomit", "voyeur", "weed", "white power", 
            "wigger", "witch", "xxx", "exploit", "hate", "violence", "bully", "abuse", 
            "harass", "threat", "illegal", "fraud", "scam", 
            "phishing", "misogyny", "misandry", "bigamy", 
            "cult", "extremist", "radical", "supremacist", 
            "xenophobe", "hazard", "injury", "risk", "unsafe", 
            "anorexia", "bulimia", "self-harm", "self-hate", 
            "addict", "addiction", "gamble", "gambling", 
            "weapons", "terrorism", "exploitation", "cult", 
            "abduct", "kidnap", "suicidal", "bomb", "gun",
            "anarchist", "bloodshed", "exploit", "intolerance",
            "kidnapping", "cultist", "sabotage", "sedition",
            "slander", "stalking", "terrorism", "torture", "violator","bastard", "bitch", "cunt", "whore", "slut", "fag", 
            "dyke", "retard", "idiot", "stupid", "moron", "douchebag",
            "n-word", "kike", "spic", "chink", "gook", "wop", 
            "cracker", "redneck", "hillbilly", "white trash", 
            "terrorist", "jihad", "isis", "kkk", "murderer", "rapist",
            "abortion", "al-Qaeda", "anal", "anarchy", "anus", "arson", 
            "asshole", "bastard", "bigot", "blasphemy", "bolshevik", 
            "bong", "booze", "cocaine", "commie", "crack", "damn", 
            "date rape", "devil-worship", "dildo", "drugs", "ecstasy", 
            "fascist", "fatso", "felching", "fellatio", "fetish", 
            "fornicate", "gangbang", "genital", "god damn", "goddamn", 
            "guro", "heroin", "homicide", "incest", "jackass", "jerk", 
            "jigaboo", "kinky", "klan", "lesbian", "masturbate", 
            "meth", "molest", "muff", "murder", "naked", "necrophilia", 
            "nipple", "opiate", "opium", "orgasm", "pedophile", "piss", 
            "poop", "prostitute", "pubic", "queer", "racist", 
            "rape", "rectum", "satanic", "scat", "semen", "sexism", 
            "sexual", "shroom", "snorting", "sodomize", "sodomy", 
            "spastic", "suicide", "terror", "threesome", "toke", 
            "torture", "toxic", "transexual", "underage", "urinate", 
            "urine", "vomit", "voyeur", "weed", "white power", 
            "wigger", "witch", "xxx", "exploit", "hate", "violence", "bully", "abuse", 
            "harass", "threat", "illegal", "fraud", "scam", 
            "phishing", "misogyny", "misandry", "bigamy", 
            "cult", "extremist", "radical", "supremacist", 
            "xenophobe", "hazard", "injury", "risk", "unsafe", 
            "anorexia", "bulimia", "self-harm", "self-hate", 
            "addict", "addiction", "gamble", "gambling", 
            "weapons", "terrorism", "exploitation", "cult", 
            "abduct", "kidnap", "suicidal", "bomb", "gun",
            "anarchist", "bloodshed", "exploit", "intolerance",
            "kidnapping", "cultist", "sabotage", "sedition",
            "slander", "stalking", "terrorism", "torture", "violator", "niga", "niger", "nigger"
        }

        # Create a regular expression pattern for filtering
        pattern = '|'.join(rf'\b{word}\b' for word in filter_dict)
        pattern2 = '|'.join(rf'{word}' for word in filter_dict)

        # Use boolean indexing to filter out rows containing offensive content
        df = df[~df['joke'].str.contains(pattern, case=False, regex=True)]
        df = df[~df['joke'].str.contains(pattern2, case=False, regex=True)]    

                  

        df.to_csv('cleaned_dataset.csv', index=False) # save cleaned dataset


#Execute the code
csv_file_path = r'C:\Users\49174\Documents\Studium - AI\Semester II\NLP - Semantic Data\Group project\fine-tuning\reddit_dadjokes.csv'
djc = DadJokesCleaner(csv_file_path)
djc.clean_data()
