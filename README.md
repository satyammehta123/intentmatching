# DOJ-Intent-Matching

**Description:** 

This repository contains code for intent classification using deep learning techniques in the field of natural language processing (NLP). The goal of this project is to develop a model that can accurately classify user queries into specific intents, enabling intelligent systems to understand and respond appropriately to user inputs.  In this project, we leverage the power of Keras, a popular deep learning library, to build a sequential model that can effectively learn and predict intents based on input queries.

**Installation:**

To run the code in this repository, you need the following dependencies:

Install Python on your system.
Install the required libraries by running the appropriate commands 
Download the code files from the repository.
Ensure that the intents.json file is present in the same directory as the code files.
Run the train.py file.


**Test**

Run the main.py script to train the chatbot model using the intents from the intents.json file.
Enter your queries when prompted by the chatbot.
To exit the chat, type "quit".

**Approach:**

Preprocessing: We begin by preprocessing the training data. This includes removing stopwords, lemmatizing words, and tokenizing sentences. The preprocessing step enhances the quality of the training data and improves the model's ability to learn meaningful representations.

Model Architecture: We construct a sequential model using Keras. The model consists of an embedding layer, an LSTM layer, a dropout layer, and a dense layer. Dropout regularizes the model to prevent overfitting, and the dense layer produces the final output probabilities for each intent category.

Evaluation: We evaluate the trained model on a validation dataset, computing metrics such as accuracy and loss. This assessment allows us to measure the model's performance and validate its ability to correctly predict intents for unseen data.

1. What information is typically included in immigration records?
2. Are there any restrictions on accessing historical travel information?
3. I need to request a tax transcript from the IRS. What are the available methods for obtaining it?
4. Can I request medical records from a previous healthcare provider if I have changed doctors?
5. Can I request disciplinary records from my personnel file?
6. Can you provide information on the organizational structure and chain of command of the U.S. Army?


Prediction: With the trained model, we can make intent predictions for user queries. By inputting a query, the model predicts the most likely intent category, enabling intelligent systems to understand and respond accordingly.
