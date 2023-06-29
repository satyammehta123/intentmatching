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

#fuzzywuzzy library used for string-matching and string comparison
from fuzzywuzzy import fuzz


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




##################################################################################################
#TRAINING

#initialise lists for training both the tokens and the intents
tokens_train = []
intents_train = []

#loop through the list of tokensXintents
for (i, j) in tokensXintents:
    
    #call embedSentence to embed tokenised sentence list from iteration
    temp = encodeSentence(i)
    
    #add the embedded sentence to list tokens_train
    tokens_train.append(temp)
        
    #call embedSentence to embed the intent from iteration too
    temp2 = encodeIntents(j)
    
    print(len(temp2))
        
    #add the embedded sentence to list intents_train
    intents_train.append(temp2)
    
    # #label variable assigns numeric label to each intent. label is later used as the target or ground truth during model training since machine learning algorithms typically expect class labels to be represented as integers. 
    # #by using the label variable, you can provide the correct intent label to the model during the training process.
    # label = intents_list.index(classification)
    
    # #add the index of the intent(classification) on the list to the intents_train list
    # intents_train.append(label)

exit()

"""""
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

#set the input_size to the length of the first element in tokens_train (broken words from queries)
input_size = len(tokens_train[0])

for i in range(len(tokens_train)):
    print(i)

#set hidden_size to 8
#juggles the line between increasing computational requirements and risking overfitting (capacity and complexity)
hidden_size = 3

#set the output_size to the length of the intents list
output_size = len(intents_list)

#define class chatDataset that represents a dataset for training a model in conversational application
#parameter is designed to be compatible with pytorch "Dataset" class, which is a standard class for handling datasets
#by inheriting from Dataset, the ChatDataset class can leverage the functionalities by the base class 
class ChatDataset(Dataset):

    # define constructor method of the class and sets up initial values for its attributes
    def __init__(self):
        
        #assigns number of samples in tokens_train to the n_samples attribute of the class. Represents the number of samples
        self.n_samples = len(tokens_train)
        
        #assigns the tokens_train dataset to the x_data attribute of the class. Represents input data of the model
        self.x_data = tokens_train
        
        #assigns intents_train dataset to the y_data attribute to the class. Represents corresponding label data for model
        self.y_data = intents_train

    #support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        
        #returns the input data (x_data) and it's corresponding label (y_data) at specified data
        return self.x_data[index], self.y_data[index]

    #returns size of dataset
    def __len__(self):
        
        #returns the total number of samples (n_samples) in dataset
        return self.n_samples


# create a new object of ChatDataset and assign it to variable dataset. Represents training dataset for your chat model
dataset = ChatDataset()

#create pytorch "dataloader" object named train_loader responsible to load the data in batches during training
#dataset specifies the ChatDataset object
#batch_size determines the number of samples to be included in each mini_batch during training
#indicates whether data should be shuffled before each epoch. Shuffling allows set to be randomly ordered, reducing bias
#specifies number of worker processes to use for data loading. 
#num_workers= 0 means data loading will be performed in main process, without using added worker processes for parallelism
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#if cuda enabled GPU is available to use, else, use cpu device.
#ensures the program utilises GPU acceleration when possible, maximising performance of the computations
#torch.device() creates a pytorch device object with the string of the device type as an argument
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#create an instance of the neural network with the input_size, hidden_size, and num_classes
#Also sets the device on which the model will be located (gpu or cpu)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#nn.CrossEntropyLoss computes the negative log-likelihood loss between the predicted probabilities and the target labels
#used during training process to calculate the loss and optimise the model paramters. 
# loss value represents the dissimilarity between the predicted class probabilities and the target labels
criterion = nn.CrossEntropyLoss()

#creates an instance of the Adam optimiser and associates it with model's parameters
#parameters are the already-defined parameters of the model and the learning rate  
#using adam optimiser allows model to undergo parameter updates during training, allowing efficient optimisation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

#for each epoch (time the dataset is passed through the model)
for epoch in range(num_epochs):
    
    for (words, labels) in train_loader:
        
        words = words.to(device)
        labels = labels.to(device)

        #perform forward pass of model on the current batch of sentences and obtains predicted outputs
        outputs = model(words)
        
        #calculate loss value based on predicted outputs and target intents
        #criterion is instance of nn.CrossEntropyLoss() function 
        #function combines softmax function and negative log-likelihood loss calculation
        loss = criterion(outputs, labels)

        #zero out (set = 0)the gradients of the model parameters before the backward pass and parameter update
        #gradients need to be updated based on the loss value back propagation
        #ensures gradients are fresh and not influenced by any previous computations
        optimizer.zero_grad()
        
        
        #loads the optimiser state as dict
        # dictState = optimizer.state.dict()
        
        #perform backward pass of the model (computes gradients of the model parameters with respect to loss value)
        #in training process, the goal is to minimise the loss function by adjusting the model parameters
        #the backward pass enables the neural network to learn from the training data and improve performance over passes 
        loss.backward()
        
        #optimiser.step() updates the model parameters based on the gradients computed suring the backward pass
        #updates the parameters of the model in the direction that reduces the loss function
        optimizer.step()

    #print training progress during each batch
    if (epoch+1) % 10 == 0:
        
        #print epoch number, batch number, and corresponding loss value
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print the final loss
print(f'final loss: {loss.item():.4f}')

#
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": intents_list
}

#save trained model file as data.pth
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
"""""