#model.py

#created by Kirk Ogunrinde on Jun 23, 2023

##################################################################################################
#IMPORTS

#torch library used for building and training neural networks
#provides wide range of tools and functionalities for efficient numerical computing and machine learning tasks

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sequence_length = sequence_length

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]  # Convert single sequence to a list

        max_len = self.sequence_length
        x = torch.tensor(x)  # Convert sequences to a tensor
        padded_x = F.pad(x, (0, max_len - x.size(1)))[:, :max_len]  # Pad and truncate sequences

        x = self.fc1(padded_x)
        x = self.relu(x)
        x = self.fc2(x)
        return x




# Define the custom dataset class
class MyDataset(Dataset):
    def __init__(self, tokens_train, intents_train):
        self.tokens_train = tokens_train
        self.intents_train = intents_train

    def __len__(self):
        return len(self.tokens_train)

    def __getitem__(self, index):
        token = self.tokens_train[index]
        intent = self.intents_train[index]
        return token, intent