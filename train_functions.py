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

#fuzzywuzzy library used for string-matching and string comparison
from fuzzywuzzy import fuzz

#sentence_transformer library allows one to create and work with sentence embeddings
#provides functionality for similarity search, clustering, classification, or transfer learning in the field of natural language processing.
from sentence_transformers import SentenceTransformer

#create an instance of the class by specifying the pre-trained model to use.
model = SentenceTransformer('nli-mpnet-base-v2')

# model = SentenceTransformer('nli-roberta-base-v2')
# model = SentenceTransformer('princeton-nlp/sup-simcse-roberta-large')
# model = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')
# model = SentenceTransformer('stsb-distilroberta-base-v2')
# model = SentenceTransformer('stsb-mpnet-base-v2')
# model = SentenceTransformer('stsb-roberta-bas')
# model = SentenceTransformer('stsb-roberta-base-v2')
# model = SentenceTransformer('stsb-roberta-large'')


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


#function to embed the sentences
#function takes in a sentence, embeds it and returns it embedded
def embedSentence(sentence):
    
    #call the encode() method of the model, which returns the corresponding sentence embeddings
    embeddedSentence = model.encode(sentence)

    print(len(embeddedSentence))

    return embeddedSentence