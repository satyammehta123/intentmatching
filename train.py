#train.py

#created by Kirk Ogunrinde on Jun 23, 2023

##################################################################################################
#IMPORTS

#from "filename" import * allows one file to access all fuctions from another file
from model import *
from train_functions import *

#json library provides functions for working with JSON data
#allows serialise (encode) python objects to JSON strings and vice-versa
import json

#numpy library provides tools and functions for working with arrays, numerical computations and linear algrbra operations
import numpy as np

#torch library used for building and training neural networks
#provides wide range of tools and functionalities for efficient numerical computing and machine learning tasks
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

#fuzzywuzzy library used for string-matching and string comparison
from fuzzywuzzy import fuzz

import time
start = time.time











##################################################################################################
#PRE-PROCESSING
#open intents.json file in read mode
with open('intents.json', 'r') as file:

    #json.load() deserialises a JSON file into a python object
    #the resulting python object could be a dict, list, string, etc based on the structure of the JSON data
    intents = json.load(file)

#initialise some lists 
intents_list = []
tokensXintents = []

#loop through each block in our intents.json file
for block in intents['intents']:
    
    #attribute variable intent to the intent in the "tag" block
    intent = block['tag']
    
    #add the intent to the intents list 
    intents_list.append(intent)
    
    #loop through each query in the "patterns" block
    for query in block['patterns']:
        
        #tokenize the query
        tokens = tokenize(query)
        
        #add a pairing of tokenised words and intent to list tokensXintents
        tokensXintents.append((tokens, intent))



#initialise lists for training both the tokens and the intents
tokens_train = []
intents_train = []

#loop through the list of tokensXintents
for (i, j) in tokensXintents:
    
    #call embedSentence to embed tokenised sentence list from iteration
    temp = encodeSentence(i)
        
    #add the embedded sentence to list tokens_train
    tokens_train.append(temp)
    
    #label variable assigns numeric label to each intent. label is later used as the target or ground truth during model training since machine learning algorithms typically expect class labels to be represented as integers. 
    #by using the label variable, you can provide the correct intent label to the model during the training process.
    label = intents_list.index(j)
    
    #add the index of the intent(classification) on the list to the intents_train list
    intents_train.append(label)




##################################################################################################
#TRAINING
#define hyper-parameters for training

#set num_epochs to 1000
#defines number of times the entire training dataset will be passed through model during training phase
#rides line between improving performance and overfitting
#monitoring the model's perormance on a validation set helps determine optimal number of epochs
num_epochs = 100

#set batch_size to 8
#determines the numnber of training samples processed in each iteration of the training loop
#using mini-batches instead of processing the entire dataset could improve training efficiency
#optimal batch size depends on the available computational resurces and characteristics of the dataset
#smaller batch sizes improves stochasticiity while larger leads to smoother gradient sizes but increased memory requirements 
batch_size = 3
 
#set learning_rate to 0.001
#controls the speed the model learns at. rides lines between faster convergence and instabilty/overshooting
learning_rate = 0.001


# Create an instance of the model
input_size = len(tokens_train[0])
hidden_size = 128
output_size = len(intents_list)


max_seq_length = max(len(seq) for seq in tokens_train)

model = MyModel(input_size, hidden_size, output_size, max_seq_length)


# Pad sequences to a fixed length
padded_dataset = []
max_seq_length = max(len(seq) for seq in tokens_train)
for seq in tokens_train:
    padded_seq = seq + [0] * (max_seq_length - len(seq))
    padded_dataset.append(padded_seq)

# Update tokens_train with padded_dataset
tokens_train = padded_dataset


# Create the dataset and data loader
dataset = MyDataset(tokens_train, intents_train)

# Define the collate function for the DataLoader
def collate_fn(batch):
    
    tokens_batch = [item[0] for item in batch]
    intents_batch = [item[1] for item in batch]

    # Pad the tokens to the length of the longest sequence
    max_len = max(len(tokens) for tokens in tokens_batch)
    padded_tokens_batch = []
    for tokens in tokens_batch:
        padded_tokens = tokens + [0] * (max_len - len(tokens))
        padded_tokens_batch.append(padded_tokens)

    intents_batch = torch.tensor(intents_batch)  # Convert intents to a tensor
    return padded_tokens_batch, intents_batch


# Create the training data loader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
# Training loop
total_steps = len(dataloader)
for epoch in range(num_epochs):
    for i, (tokens, intents) in enumerate(dataloader):
        
        # Forward pass
        outputs = model(tokens)
        loss = criterion(outputs, intents)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

print("Training finished.")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pt')

print("Trained model saved.")






##################################################################################################
# VALIDATION

# Load the intents2.json file for validation
with open('intents2.json', 'r') as file:
    intents2 = json.load(file)

tokensXintents2 = []

for block in intents2['intents']:
    intent = block['tag']
    for query in block['patterns']:
        tokens = tokenize(query)
        tokensXintents2.append((tokens, intent))

tokens_validate = []
intents_validate = []

for (i, j) in tokensXintents2:
    temp = encodeSentence(i)
    tokens_validate.append(temp)
    intents_validate.append(j)

# Create the validation dataset and data loader
validation_dataset = MyDataset(tokens_validate, intents_validate)

# 
print("validation")

# Define the collate function for the validation DataLoader
def collate_fn(batch):
    tokens_batch = [item[0] for item in batch]
    intents_batch = [item[1] for item in batch]

    # Pad the tokens to the length of the longest sequence
    max_len = max(len(tokens) for tokens in tokens_batch)
    padded_tokens_batch = []
    for tokens in tokens_batch:
        padded_tokens = tokens + [0] * (max_len - len(tokens))
        padded_tokens_batch.append(padded_tokens)

    intents_batch = torch.tensor(intents_batch)  # Convert intents to a tensor
    return padded_tokens_batch, intents_batch

# Create the validation data loader
validation_dataloader = DataLoader(validation_dataset, batch_size=1, collate_fn=collate_fn)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Evaluate the model using the DataLoader
correct = 0
total = 0

with torch.no_grad():
    for tokens, intents in dataloader:
        outputs = model(tokens)
        _, predicted = torch.max(outputs, dim=1)
        total += intents.size(0)
        correct += (predicted == intents).sum().item()

accuracy = correct / total
print('Accuracy: {:.2%}'.format(accuracy))