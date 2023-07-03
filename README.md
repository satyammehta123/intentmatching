# DOJ-Intent-Matching

**Description:** 

This repository contains code for intent classification using deep learning techniques in the field of natural language processing (NLP). The goal of this project is to develop a model that can accurately classify user queries into specific intents, enabling intelligent systems to understand and respond appropriately to user inputs.  In this project, we leverage the power of Keras, a popular deep learning library, to build a sequential model that can effectively learn and predict intents based on input queries.

**Installation:**

To run the code in this repository, you need the following dependencies:

Python (3.6 or higher)
NLTK (Natural Language Toolkit)
Keras (2.4.3 or higher)
TensorFlow (2.4.1 or higher)
PyTorch (1.7.0 or higher)

**Usage:**

To interact with the chatbot, follow these instructions:

Run the main.py script to train the chatbot model using the intents from the intents.json file.
Enter your queries when prompted by the chatbot.
To exit the chat, type "quit".

**Approach:**
Preprocessing: We begin by preprocessing the training data. This includes removing stopwords, lemmatizing words, and tokenizing sentences. The preprocessing step enhances the quality of the training data and improves the model's ability to learn meaningful representations.

Model Architecture: We construct a sequential model using Keras. The model consists of an embedding layer, an LSTM layer, a dropout layer, and a dense layer. Dropout regularizes the model to prevent overfitting, and the dense layer produces the final output probabilities for each intent category.

Evaluation: We evaluate the trained model on a validation dataset, computing metrics such as accuracy and loss. This assessment allows us to measure the model's performance and validate its ability to correctly predict intents for unseen data.

Prediction: With the trained model, we can make intent predictions for user queries. By inputting a query, the model predicts the most likely intent category, enabling intelligent systems to understand and respond accordingly.