
#created by Kirk Ogunrinde on the 3rd of July, 2023


##################################################################################################
#IMPORTS
from main_functions import preprocess_sentence

import json
import torch
import torch.nn as nn
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize



##################################################################################################
#load training dataset from JSON file
with open('intents.json', 'r') as file:
    intents_train = json.load(file)

print()
print("...just loaded the traiing dataset from the intents.json file")
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

#define neural network model
class IntentClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IntentClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


##################################################################################################
#sentence Encoding for training data

#defining the model_name to use for training
model_name = 'bert-base-uncased'

# model_name = 'nli-mpnet-base-v2'
# model_name = 'nli-roberta-base-v2'
# model_name = 'princeton-nlp/sup-simcse-roberta-large'
# model_name = 'princeton-nlp/unsup-simcse-roberta-large'
# model_name = 'stsb-distilroberta-base-v2'
# model_name = 'stsb-mpnet-base-v2'
# model_name = 'stsb-roberta-bas'
# model_name = 'stsb-roberta-base-v2'
# model_name = 'stsb-roberta-large'

model = SentenceTransformer(model_name)

#encode the pre-processed sentences
encoded_sentences_train = model.encode(preprocessed_sentences_train)

#make the encoded sentences standardised using the normalize() function
normalized_encoded_sentences_train = normalize(encoded_sentences_train)

#encode the intents
encoded_intents_train = model.encode(intents_list_train)

#make the encoded intents standardised using the normalize() function
normalized_encoded_intents_train = normalize(encoded_intents_train)

print()
print("...just finished encoding sentences")
print()


##################################################################################################
#convert training lists to tensors
encoded_sentences_train = torch.tensor(normalized_encoded_sentences_train)
intents_list_train = torch.tensor(normalized_encoded_intents_train)



##################################################################################################
# define the intent classifier model
input_size = normalized_encoded_sentences_train.shape[1]
num_classes = len(set(intents_list_train))
classifier = IntentClassifier(input_size, num_classes)


##################################################################################################
#set training device to CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = classifier.to(device)


# Convert numpy arrays to PyTorch tensors
normalized_encoded_sentences_train = torch.tensor(normalized_encoded_sentences_train).to(device)
normalized_encoded_intents_train = torch.tensor(normalized_encoded_intents_train).to(device)


##################################################################################################
#define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


##################################################################################################
#train the intent classifier
num_epochs = 10
batch_size = 32
num_batches = len(normalized_encoded_sentences_train) // batch_size

for epoch in range(num_epochs):
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        inputs = normalized_encoded_sentences_train[start_idx:end_idx]
        targets = normalized_encoded_intents_train[start_idx:end_idx]

        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()



##################################################################################################
#load validation dataset
with open('intent2.json', 'r') as file:
    intents_val = json.load(file)


##################################################################################################
#preprocess and encode the validation data
sentences_list_val = []

for block in intents_val['intents']:
    intent = block['tag']
    sentences = block['patterns']
    
    for query in sentences:
        preprocessed_query = preprocess_sentence(query)
        encoded_query = model.encode([preprocessed_query])[0]
        sentences_list_val.append((query, encoded_query))


##################################################################################################
#convert validation data to tensors
encoded_sentences_val = torch.tensor([encoded_query for _, encoded_query in sentences_list_val])
encoded_sentences_val = encoded_sentences_val.to(device)


##################################################################################################
#perform prediction on the validation data
classifier.eval()
with torch.no_grad():
    outputs = classifier(encoded_sentences_val)
    _, predicted = torch.max(outputs, 1)
    predicted_intents = predicted.cpu().numpy()


##################################################################################################
#print validation classification report
y_true_val = [intent for _, intent in sentences_list_val]
print("Validation Classification Report:")
print(classification_report(y_true_val, predicted_intents))


##################################################################################################
#user Query Prediction
user_query = input("Enter your query: ")
preprocessed_query = preprocess_sentence(user_query)
encoded_query = model.encode([preprocessed_query])[0]
encoded_query = torch.tensor(encoded_query).unsqueeze(0).to(device)

classifier.eval()
with torch.no_grad():
    output = classifier(encoded_query)
    predicted_intent = torch.argmax(output).item()

print("Query:", user_query)
print("Predicted Intent:", predicted_intent)