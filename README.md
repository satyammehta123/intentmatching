# DOJ-Intent-Matching

**Description:** 

This repository contains code for intent classification using deep learning techniques in the field of natural language processing (NLP). The goal of this project is to develop a model that can accurately classify user queries into specific intents, enabling intelligent systems to understand and respond appropriately to user inputs.  In this project, we leverage the power of Keras, a popular deep learning library, to build a sequential model that can effectively learn and predict intents based on input queries.
This project uses a sequential model implemented using the Keras library. Here's an explanation of the model architecture:

1. Embedding Layer
The first layer in the model is an Embedding layer. It is responsible for learning and representing word embeddings.
The embedding layer takes the input sequence of words and maps each word to a dense vector representation.
The embedding layer helps capture the semantic meaning of words and their relationships in the input sequence.
It uses the total_words parameter to determine the input size and max_sequence_length to specify the input length.

2. LSTM Layer
The next layer is an LSTM (Long Short-Term Memory) layer.
LSTM is a type of recurrent neural network (RNN) layer that is well-suited for sequence processing tasks.
The LSTM layer processes the input sequence and learns to capture long-term dependencies and temporal patterns.
It uses the 64 parameter to specify the number of hidden units (also known as the LSTM cell size).

3. Dense Layer with SoftMax Activation Function
The Dense layer maps the LSTM layer's output to the number of intents present in the training data.
It produces a probability distribution over the intents, indicating the likelihood of each intent.
The softmax activation ensures that the predicted probabilities sum up to 1.
Model Compilation:

4. Compilation
After defining the layers, the model is compiled using the compile() method.
The categorical_crossentropy loss function is used since this is a multi-class classification problem.
The adam optimizer is chosen for training the model, which adapts the learning rate during training.
Additionally, the accuracy metric is specified to monitor the model's performance during training.
Model Training:

5. Training
The model is trained using the fit() method.
The training data (X_train and y_train) is provided, along with the number of epochs to train for (10 in this case).
The validation data (X_val and y_val) is passed to the validation_data parameter to evaluate the model's performance during training.
Model Evaluation:

6. Evaluation
After training, the model's performance is evaluated on the validation set using the evaluate() method.
The accuracy and loss on the validation set are computed and displayed.
Prediction and User Input:

7. User Query
The trained model is then used to make predictions on user input.
The user's input is tokenized, padded, and passed to the model's predict() method.
The model outputs probabilities for each intent.
The predicted intents are sorted in descending order of probabilities, and the top intents are displayed to the user.


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