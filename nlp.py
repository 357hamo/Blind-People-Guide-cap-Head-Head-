from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np


def tokenize(sentence):
    words = word_tokenize(sentence)
    return words



def stem(words, pos='v'):
    stemming = WordNetLemmatizer()
    stemmer = [stemming.lemmatize(token.lower()) for token in words]
    return stemmer

def bag_of_word(words, all_words):
    word_stemming = stem(words)
    bag = np.zeros(len(all_words),dtype='f')
    for idx,word in enumerate(all_words):
        if word in word_stemming:
            bag[idx] = 1
    return bag


if __name__ == "__main__":
    all_words = ['hi', 'hello', 'i', 'you', 'bye', 'are','thank', 'cool', 'how']
    sentence = "hello,How are you? and I'm fine thank you"
    tokenized = tokenize(sentence)
    print(tokenized)
    stemm = stem(tokenized)
    print(stemm)
    bag = bag_of_word(tokenized, all_words)
    print(bag)
