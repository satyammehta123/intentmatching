
#created by Kirk Ogunrinde on July 3rd, 2023

##################################################################################################
#IMPORTS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from fuzzywuzzy import fuzz



##################################################################################################
#preprocess sentences by removing stopwords, lemmatising, and tokenising
def preprocess_sentence(sentence):

    #set stopword-removing language to English
    stop_words = set(stopwords.words('english'))
    
    #initialise word new lemmatiser
    lemmatizer = WordNetLemmatizer()

    #split sentences to tokens and convert to lowercase
    tokens = word_tokenize(sentence.lower())
    
    #remove stopwords and punctuations
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    #lemmatise tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # #
    
    # temp = []
    # for i in lemmatized_tokens:
    #     if fuzz.ratio(i) > 70:         
    #         temp.append(i)
    
        
    #return the lemmatised tokens with spaces to resemble normal sentences 
    return ' '.join(lemmatized_tokens)