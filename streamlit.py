
# created by Kirk Ogunrinde on July 4th, 2023


##################################################################################################
#IMPORTS
import streamlit as st
from main_functions import preprocess_sentence
import os
import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, Dropout
import tensorflow as tf
import torch
from fuzzywuzzy import fuzz
from fuzzywuzzy import process





##################################################################################################

# Set Streamlit app title
st.title("Intent Matching App")

# Load training dataset from JSON file
with open('intents.json', 'r') as file:
    intents_train = json.load(file)

# Preprocess training data
intents_list_train = []
sentences_list_train = []

for block in intents_train['intents']:
    temp = block['tag']
    for query in block['patterns']:
        sentences_list_train.append(query)
        intents_list_train.append(temp)

preprocessed_sentences_train = []
for sentence in sentences_list_train:
    preprocessed_sentences_train.append(preprocess_sentence(sentence))

# Tokenize sentences for the Keras model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_list_train)
total_words = len(tokenizer.word_index) + 1

# Format data for model processing
sequences = tokenizer.texts_to_sequences(sentences_list_train)
max_sequence_length = max([len(x) for x in sequences])

# Get unique intents from the training data
unique_intents = list(set(intents_list_train))

# Load pre-trained model
load_model_file = st.text_input("Enter the filename of the pre-trained model:")
load_model_path = os.path.join(os.getcwd(), load_model_file)
model = None

if os.path.isfile(load_model_path):
    model = load_model(load_model_path)
    st.write("Pre-trained model loaded from '{}'.".format(load_model_path))
else:
    st.write("File '{}' does not exist. Please provide a valid filename.".format(load_model_file))
    st.stop()

# User input
user_input = st.text_input("Enter a query:")

if user_input:
    # Tokenize and pad user input
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length)

    # Get prediction
    prediction = model.predict(user_padded_sequence)[0]
    sorted_indices = np.argsort(prediction)[::-1]  # Sort indices in descending order

    # Display predicted intents and probabilities
    st.write('Predicted Intents:')
    for i in sorted_indices:
        intent = unique_intents[i]
        probability = prediction[i] * 100
        st.write('- {}: {:.2f}%'.format(intent, probability))
