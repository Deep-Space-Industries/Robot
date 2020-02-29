import numpy as np
import random

# Sets random weights between 0 and 1 for each element in a matrix
def randomWeights(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = random.random()
    return x

# Applying sigmoid function to a value x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, activationFunction, learningRate):
        self.inputnodes = inputnodes
        self.n_hiddenLayers = len(hiddennodes)
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes

        self.weightsIH = randomWeights(np.zeros((self.inputnodes, self.hiddennodes[0]), dtype=float))
        self.weightsHH = []
        self.biasIHH = []
        for l in range(self.n_hiddenLayers-1):
            w = randomWeights(np.zeros((self.hiddennodes[l], self.hiddennodes[l+1]), dtype=float))
            self.weightsHH.append(w)
            b = randomWeights(np.zeros((1, self.hiddennodes[l]), dtype=float))
            self.biasIHH.append(b)
        b = randomWeights(np.zeros((1, self.hiddennodes[-1]), dtype=float))
        self.biasIHH.append(b)
        self.weightsHO = randomWeights(np.zeros((self.hiddennodes[-1], self.outputnodes), dtype=float))
        self.biasHO = randomWeights(np.zeros((1, self.outputnodes), dtype=float))

        self.activationFunction = activationFunction
        self.learningRate = learningRate

    def forwardPropagation(self, x):
        # Result: 1st Hidden Layer
        a1 = np.dot(self.weightsIH.T, x)
        a1 = np.add(a1, self.biasIHH[0])

        # Activation function: 1st Hidden Layer
        for num in range(self.hiddennodes[0]):
            a1[0][num] = self.activationFunction(a1[0][num])
        # Result: n Hidden Layer
        for l in range(self.n_hiddenLayers-1):
            a1 = np.dot(self.weightsHH[l].T, a1[0])
            a1 = np.add(a1, self.biasIHH[l+1])

            # Activation function: n Hidden Layer
            for num in range(self.hiddennodes[l+1]):
                a1[0][num] = self.activationFunction(a1[0][num])

        # Result: Output Layer
        a1 = np.dot(self.weightsHO.T, a1[0])
        a1 = np.add(a1, self.biasHO)

        # Activation function: Output Layer
        for num in range(self.outputnodes):
            a1[0][num] = self.activationFunction(a1[0][num])

        return a1

    def print(self):
        print("Input -> 1 . Hidden")
        print(self.weightsIH)
        print("Bias Input -> 1 . Hidden")
        print(self.biasIHH[0])
        for i in range(len(self.weightsHH)):
            print((i+1),". Hidden ->", (i+2), ". Hidden")
            print(self.weightsHH[i])
            print((i+1),". Bias Hidden ->", (i+2), ". Hidden")
            print(self.biasIHH[i+1])
        print((len(self.weightsHH)+1),". Hidden -> Output")
        print(self.weightsHO)
        print((len(self.weightsHH)+1),". Bias Hidden -> Output")
        print(self.biasHO)

nn = NeuralNetwork(12,[4],2, sigmoid, 0.1)
input = np.array([[5,10,9,3,12,8,5,10,9,3,12,8]])
output = nn.forwardPropagation(input[0])
nn.print()

print("Output")
print(output)
