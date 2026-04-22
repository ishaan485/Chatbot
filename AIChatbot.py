import nltk
import random
import string
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmer = WordNetLemmatizer()

corpus = """
hello hi hey
how are you i am fine thank you
what is ai artificial intelligence is the simulation of human intelligence
what is machine learning machine learning is a subset of ai
who are you i am an ai chatbot created for learning purposes
bye goodbye see you later
"""

sent_tokens = nltk.sent_tokenize(corpus)
word_tokens = nltk.word_tokenize(corpus)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation))))

def response(user_input):
    sent_tokens.append(user_input)
    TfidfVec = nltk.TextCollection([normalize(sent) for sent in sent_tokens])
    sims = [TfidfVec.tf_idf(user_input, sent) for sent in sent_tokens]
    idx = sims.index(max(sims))
    if max(sims) == 0:
        return "I didn't understand that. Can you rephrase?"
    else:
        return sent_tokens[idx]

def chatbot():
    print("Chatbot is running... Type 'bye' to exit")
    while True:
        user_input = input().lower()
        if user_input in ['bye', 'exit', 'quit']:
            print("Goodbye! Have a nice day.")
            break
        else:
            print(response(user_input))

chatbot()