
#created by Kirk Ogunrinde on the 3rd of July, 2023



##################################################################################################
#IMPORTS
from main_functions import preprocess_sentence

import os
import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


##################################################################################################
#load training dataset from JSON file
with open('intents.json', 'r') as file:
    intents_train = json.load(file)

print()
print("...just loaded the training dataset from the intents.json file")
print()



##################################################################################################
#preprocess training data
intents_list_train = []
sentences_list_train = []

#looop through the intent.json file
for block in intents_train['intents']:
    
    #temp equals the intent (IRS Records, Medical Records, etc.)
    temp = block['tag']
    
    #add each sentence to the sentences_list_train and add its intent to the intents_list_train so we can have parallel lists
    for query in block['patterns']:
        sentences_list_train.append(query)
        intents_list_train.append(temp)

#preprocess each sentence (remove stopwords, lemmatise, tokenise)
preprocessed_sentences_train = []
for sentence in sentences_list_train:
    preprocessed_sentences_train.append(preprocess_sentence(sentence))


print()
print("...just finished preprocessing the training sentences")
print()


##################################################################################################
#tokenize sentences for keras model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_list_train)
total_words = len(tokenizer.word_index) + 1



##################################################################################################
#format data for keras model processing 

#convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences_list_train)

# Padding sequences
max_sequence_length = max([len(x) for x in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Get unique intents from the training data
unique_intents = list(set(intents_list_train))

# Convert intents to one-hot encodings
labels = np.zeros((len(sentences_list_train), len(unique_intents)))
for i, intent in enumerate(intents_list_train):
    labels[i, unique_intents.index(intent)] = 1

#split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.7, random_state=42)


print()
print("...just finished converting the intents to one-hot encodings and splitting the dataset")
print()



##################################################################################################
#model definition
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_length))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(len(set(unique_intents)), activation='softmax'))

# Adjust the learning rate
learning_rate = 0.001

# Compile the model with the modified learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


train_model = input("Do you want to train a new model? (y/n): ")

if train_model.lower() == 'y':
    
    ##################################################################################################
    #model training
    # Train the model on the full training set
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    # Compute precision and loss on the full dataset
    _, accuracy = model.evaluate(X_train, y_train)
    print('Training Set Accuracy: {:.2f}%'.format(accuracy * 100))

    predictions = model.predict(X_val)
    predicted_intents = [unique_intents[np.argmax(pred)] for pred in predictions]
    true_intents = [unique_intents[np.argmax(label)] for label in y_val]

    
    # Print predicted intents and true intents side by side
    for predicted, true in zip(predicted_intents, true_intents):
        print("Predicted: ", predicted, "True: ", true)

    
    # Calculate precision
    precision = sum([1 for pred, true in zip(predicted_intents, true_intents) if pred == true]) / len(predicted_intents)
    print('Precision: {:.2f}%'.format(precision * 100))
    print()

    # # Save the trained model
    # model.save('trained_model.h5')
    # print("Trained model saved as 'trained_model.h5'.")
       
    # Convert the .h5 model to SavedModel format
    tf.saved_model.save(model, 'saved_model')

    print("Trained model saved as 'saved_model'.")
    
    
    
    
    
    ##################################################################################################
    # VALIDATION



    ##################################################################################################
    #load training dataset from JSON file
    with open('intents2.json', 'r') as file:
        intents_valid = json.load(file)

    print()
    print("...just loaded the validation dataset from the intents2.json file")
    print()



    ##################################################################################################
    #preprocess validation data
    val_intents = []
    val_sentences = []

    #looop through the intent.json file
    for block in intents_valid['intents']:
        
        #temp equals the intent (IRS Records, Medical Records, etc.)
        temp = block['tag']
        
        #add each sentence to the sentences_list_train and add its intent to the intents_list_train so we can have parallel lists
        for query in block['patterns']:
            val_sentences.append(query)
            val_intents.append(temp)

    #preprocess each sentence (remove stopwords, lemmatise, tokenise)
    preprocessed_sentences_valid = []
    for sentence in val_sentences:
        preprocessed_sentences_valid.append(preprocess_sentence(sentence))


    print()
    print("...just finished preprocessing the validation sentences")
    print()




    ##################################################################################################
    # tokenize and pad validation sentences
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length)

    # Convert validation intents to one-hot encodings
    val_unique_intents = list(set(val_intents))

    # Convert validation intents to one-hot encodings
    val_labels = np.zeros((len(val_sentences), len(val_unique_intents)))
    for i, intent in enumerate(val_intents):
        val_labels[i, val_unique_intents.index(intent)] = 1


    print()
    print("...just finished converting the validation intents into one-hot encodings and padding")
    print()



    ##################################################################################################
    #compute accuracy and loss on the validation set
    val_loss, val_accuracy = model.evaluate(val_padded_sequences, val_labels)
    print('Validation Accuracy: {:.2f}%'.format(val_accuracy * 100))
    print('Validation Loss:', val_loss)
    print()



    ##################################################################################################
    # Validation on user input

    

    while True:
        user_input = input('Enter a query (or enter "exit" to quit): ')
        if user_input.lower() == 'exit':
            break

        # Tokenize and pad user input
        user_sequence = tokenizer.texts_to_sequences([user_input])
        user_padded_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length)

        # Get prediction
        prediction = model.predict(user_padded_sequence)[0]
        sorted_indices = np.argsort(prediction)[::-1]  # Sort indices in descending order

        # Print intents and probabilities
        print('Predicted Intents:')
        for i in sorted_indices:
            intent = unique_intents[i]
            probability = prediction[i] * 100
            print('- {}: {:.2f}%'.format(intent, probability))


    
else:
    
    # load_model_file = input("Enter the filename of the pre-trained model: ")
    # load_model_path = os.path.join(os.getcwd(), load_model_file)

    # if os.path.isfile(load_model_path):
    #     model = load_model(load_model_path)
    #     print("Pre-trained model loaded from '{}'.".format(load_model_path))
        
    # else:
    #     print("File '{}' does not exist. Please provide a valid filename.".format(load_model_file))
    #     exit()   
    
    
    load_model_file = input("Enter the filename of the pre-trained model: ")
    load_model_path = os.path.join(os.getcwd(), load_model_file)

    if os.path.isdir(load_model_path):
        model = tf.saved_model.load(load_model_path)
        print("Pre-trained model loaded from '{}'.".format(load_model_path))

    else:
        print("Directory '{}' does not exist. Please provide a valid directory containing the SavedModel.".format(load_model_file))
        exit()


    ##################################################################################################
    # Validation on user input


    while True:
        user_input = input('Enter a query (or enter "exit" to quit): ')
        if user_input.lower() == 'exit':
            break

        # Tokenize and pad user input
        user_sequence = tokenizer.texts_to_sequences([user_input])
        user_padded_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length)

        # Get prediction
        prediction = model.predict(user_padded_sequence)[0]
        sorted_indices = np.argsort(prediction)[::-1]  # Sort indices in descending order

        # Print intents and probabilities 
        print('Predicted Intents:')
        for i in sorted_indices:
            intent = unique_intents[i]
            probability = prediction[i] * 100
            print('- {}: {:.2f}%'.format(intent, probability))