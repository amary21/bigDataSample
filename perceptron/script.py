import numpy as np
import pandas as pd
import sys
import csv

# Reading data file
data = pd.read_csv("datafiles.csv")
# input is a two dimensional list, each list contains a row
inputs = np.array(data[['x1', 'x2', 'x3', 'x4']])

# output is a 2d array, each subarray is a output row
output = np.array(data[['y']])


# Calss for perceptron
class Perceptron(object):
    """Implements a perceptron, learning,testing and activation
    function
    """

    # constructer
    def __init__(self, numberOfRows, lrnRt=1, epochs=30):
        # Additind 1 for threshold or Bias value
        # And All Initialize with Zero
        self.W = np.zeros(numberOfRows+1)
        self.epochs = epochs
        # learning rate
        self.lr = lrnRt

    # Activation funtion
    def Activation(self, x):
        return 1 if x >= 0 else 0

    # predicter funtion
    def predict(self, input):
        input = np.insert(input, 0, 1)
        sum = 0
        for i in range(len(input)):
            sum = sum+((self.W[i])*(input[i]))

        ActivationOutput = self.Activation(sum)
        return ActivationOutput

    # fiting weights values
    def fit(self, X, actualOutput):
        # for loop number of Epoches
        for epoch in range(self.epochs):
            print("========================================No of Epoches========================================> ", epoch)
            print("Updated Weights: ", self.W)
            # Loop for number of columns
            for i in range(actualOutput.shape[0]):
                predicted = self.predict(X[i])
                # error calculation
                err = actualOutput[i] - predicted
                self.W = self.W + self.lr * err * np.insert(X[i], 0, 1)

# Creating object
perceptron = Perceptron(numberOfRows=4)


def learn():
    perceptron.fit(inputs, output)
    print("Final Weights", perceptron.W)


if sys.argv[1] == "--learn":
    learn()

elif sys.argv[1] == "--test":
    learn()
    # testing starts
    print("=================================================Testing start here=================================================")
    for i in range(output.shape[0]):
        prd = perceptron.predict(inputs[i])
        print("inputs: ", inputs[i], "output: ",
              output[i], "predicteed: ", prd)
else:
    print("No argument are given")
