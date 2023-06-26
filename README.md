# DOJ-Intent-Matching

**Description:** 

This project implements a chatbot using Python and PyTorch. The chatbot is trained using intents from a JSON file and utilizes a bag-of-words approach along with a neural network model to classify user queries and provide appropriate responses.

**Installation**

To set up and run the project, please follow these steps:

Install Python on your system.
Install the required libraries by running the following command: pip install numpy 
pip install nltk
pip install torch.
Download the code files from the repository.
Ensure that the intents.json file is present in the same directory as the code files.

**Usage**

To interact with the chatbot, follow these instructions:

Run the train.py script to train the chatbot model using the intents from the intents.json file.
After training, run the chat.py script.
Enter your queries when prompted by the chatbot.
To exit the chat, type "quit".

**Data**

The chatbot utilizes the intents.json file to train the model. This file contains intents and associated patterns for user queries. Each intent consists of a tag and a set of patterns. Users can add new intents or patterns to the intents.json file following the provided format.

During data preprocessing, the patterns are tokenized using NLTK's word tokenizer. The tokens are then stemmed using the Porter stemming algorithm to reduce words to their root form. A bag-of-words representation is created by mapping each token to a unique index. This representation is used as input to the neural network model for training and inference.
