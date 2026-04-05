# Overview

The purpose of this project was to create a simple one layer, two node neural network from scratch. 
Ultimately it was created to understand the way that the weights and biases form matrices and how they are handled in code. Another goal was to understand the 
gradient step by calculating the derivatives manually, and ensuring that the code was properly implemented. This code would ideally serve as a supplement to the NBA prediction repo.
The plan would be to use this neural network to train a model capable of making predictions and would be used alongside the other algorithms.

After completing the one layer, two node model and seeing that it would train the weights and biases properly, this code was then developed to accomodate n-nodes and n-outputs. 
It is still currently limited to one layer, but it has proven to be effective at training models on relatively simple datasets.

Users can experiment with different learning rates, numbers of nodes, and epochs (basically one time through, 
stepping forward and back-propogating the gradient) to try get the best fit to the data.

This program is also for the training portion of the machine learning process, another program with multiple hidden layers that will accomodate training and testing is currently in the works.

## Features
  - Import n-dimensional datasets and use a neural network to train the model
  - Trains models with less computational requirements and can be effective for basic training
  - Variable number of nodes, learning rate, and epochs
  - Captures the parameters for the lowest error and stores them in a file
  - Graphs will open at 25%, 50%, 75%, and one again at the end to visualize the training progress

## Required
  - Python 3.x
  - Numpy
  - matplotlib

## Getting Started
  - git clone https://github.com/PulpCanMoveBaby/Homemade_Neural_Network.git
  - Make sure necessary libs are installed
  - Open IDE with cloned directory
  - Open nn_onelayer_oneoutput.py or nn_one_layer_threeoutput.py
  - Update user inputs, labels(observed outputs), learning rate, nodes, and trials(epochs)
      -make sure that you have the same number of inputs and outputs
        - During training, the inputs are the "question" and the outputs are the "answer"
  -Run the file
 
