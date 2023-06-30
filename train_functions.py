# train_functions.py

#created by Kirk Ogunrinde on Jun 23, 2023

##################################################################################################
#IMPORTS

#numpy library provides tools and functions for working with arrays, numerical computations and linear algrbra operations
import numpy as np

#nltk library allows for various text-processing functionality
#provides various functionalities tasks such as tokenization, stemming, lemmatization, part-of-speech tagging,etc
import nltk
from nltk.stem import WordNetLemmatizer

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



#transformers fuction to perform encoding to dimensionality of 768
def encode_sentence(sentence):
    
    # Encode a single sentence using the Sentence Transformers model
    encoded = model.encode(sentence)
        
    return encoded



#function to embed the sentences
#function takes in one tokenised sentence list, encodes it and returns another list with the list encoded
def encodeSentence(list_of_one_tokenised_sentence):
    
    # Create the list of punctuations to ignore
    ignore_words = {'?', '.', '!'}
    
    # Create a new list to hold the modified words
    modified_words = []
    
    # Create an instance of the WordNetLemmatiser
    lemmatizer = WordNetLemmatizer()
    
    # Iterate over each word in the tokenized sentence
    for i in list_of_one_tokenised_sentence:
        
        # If the word is not a punctuation, proceed with modifications
        if i not in ignore_words:
            
            # lemmatise the word and convert it to lowercase
            temp = lemmatizer.lemmatize(i.lower())
            
            # Add the modified word to the list
            modified_words.append(temp)
    
    # Encode the modified words using the Sentence Transformers model
    encoded_words = [encode_sentence(word) for word in modified_words]
    
    return encoded_words