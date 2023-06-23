# train_functions.py

#created by Kirk Ogunrinde on Jun 23, 2023

##################################################################################################
#IMPORTS

#numpy library provides tools and functions for working with arrays, numerical computations and linear algrbra operations
import numpy as np

#nltk library allows for various text-processing functionality
#provides various functionalities tasks such as tokenization, stemming, lemmatization, part-of-speech tagging,etc
import nltk
from nltk.stem.porter import PorterStemmer



##################################################################################################
#FUNCTION DEFINITIONS

#function that tokenises a sentence
# function takes in a sentence, and returns a list with the words from the sentence, broken up  
def tokenize(sentence):
   
    #tokenise the sentence and returns a list of the words 
    return nltk.word_tokenize(sentence)


#function that stems word
#function takes in a word, stems it, and returns the lowercased and stemmed version of it
def stem(word):
    
    #initialise an instance of the PorterStemmer
    stemmer = PorterStemmer()
    
    #stems the word and convert it to lower case
    return stemmer.stem(word.lower())


#function that creates a bag of words representation of a sentence
# function takes in tokenised sentence list and huge list of words and returns bag of words representation of the tokenised sentence
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    #stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    #initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        
        if w in sentence_words: 
            
            bag[idx] = 1

    return bag