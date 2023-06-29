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



#transformers fuction to perform encoding to dimensionality of 768
def encode_sentence(sentence):
    
    # Encode a single sentence using the Sentence Transformers model
    encoded = model.encode(sentence)
        
    return encoded



#function to embed the sentences
#function takes in one tokenised sentence list, encodes it and returns another list with the list encoded
def encodeSentence(list_of_one_tokenised_sentence):
    
    #create return list temp
    temp = []
    
    #create list of punctuations to ignore
    ignore_words = ['?', '.', '!']
    
    #stem word and convert to lowercase
    stemmer = PorterStemmer()

    #for each word in the list
    for i in list_of_one_tokenised_sentence:
    
        # if the word is not a punctuation, then we can go ahead and encode it
        if i not in ignore_words:
            
            # stem
            i = stemmer.stem(i)   
            
            # convert to lowercase
            i = i.lower()
            
            # encode word
            i = encode_sentence(i)
            
            # add it to the temp list
            temp.append(i)
            
    return temp


# function to embed intents
# function takes in an intent, lowercases it and returns another list with the intent encoded
def encodeIntents(a_list_that_holds_the_intent_text):
    
    # new list temp 
    temp = []
    
    # for each word in intent
    for i in a_list_that_holds_the_intent_text:
        
        # convert to lowercase
        i = i.lower()
        
        # encode word
        i = encode_sentence(i)
        
        # add to list temp
        temp.append(i)
        
    return temp 