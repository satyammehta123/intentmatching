#model.py

#created by Kirk Ogunrinde on Jun 23, 2023

##################################################################################################
#IMPORTS

#torch library used for building and training neural networks
#provides wide range of tools and functionalities for efficient numerical computing and machine learning tasks
import torch.nn as nn

# class Neuralink serves as a subclass of nn.Module. 
# In pytorch, defining a neural network involves creating a class that inherits from nn.Module
class NeuralNet(nn.Module):
    
    #define constructor with parameters self, input_size, and hidden_size, and num_classes 
    #self references the object being created allows one access the attributes and methods within the constructor
    #input_size refers to the size of the input data the network expects
    #hidden_size represents the number of neurons or units in the hidden layers of a neural network
    #num_classes defines the number of classes or categories in the classification task the network is built for
    def __init__(self, input_size, hidden_size, num_classes):
        
        #initialise the NeuralNet object
        super(NeuralNet, self).__init__()
        
        #define the layers and activation function of the neural network within the NeuralNet class
        #nn.Liner() function takes the input and output size as paramters and creates a linear model with the dimensions 
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        
        #self.relu is an instance of the nn.ReLU() function which represents the rectified linear unit activation function
        #it introduces non-linearity into the network by applying an element-wide threshholding function
        self.relu = nn.ReLU()


    #function defines the forward pass of the neural network within the neuralnet class
    #the input tensor x is passed as an argument and is passed through each layer sequentially.
    # the output of the layer is returned as the final output of the forward pass 
    def forward(self, x):
        
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        
        #no activation because we use the CrossEntropyLoss later
        return out