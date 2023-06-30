# chat.py

import json
import torch
import torch.nn as nn
from fuzzywuzzy import fuzz
from train_functions import *
from model import MyModel

# Load the intents.json file
with open('intents.json', 'r') as file:
    intents = json.load(file)

intents_list = []
tokensXintents = []

for block in intents['intents']:
    intent = block['tag']
    intents_list.append(intent)
    for query in block['patterns']:
        tokens = tokenize(query)
        tokensXintents.append((tokens, intent))

tokens_train = []
intents_train = []

for (i, j) in tokensXintents:
    temp = encodeSentence(i)
    tokens_train.append(temp)
    intents_train.append(j)

# Load the trained model
input_size = len(tokens_train[0])
hidden_size = 128
output_size = len(intents_list)

model = MyModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('trained_model.pt'))
model.eval()

# Function to predict the intent for a query
def predict_intent(query):
    tokens = tokenize(query)
    encoded_tokens = encodeSentence(tokens)
    inputs = torch.tensor(encoded_tokens).unsqueeze(0).float()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    intent = intents_list[predicted.item()]
    return intent

# Chat loop
print("Welcome to the chat bot! Type 'quit' to exit.")
while True:
    user_query = input("User: ")
    if user_query == "quit":
        print("Chat ended.")
        break
    intent = predict_intent(user_query)
    print(f"Bot: Intent: {intent}")
