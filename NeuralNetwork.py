import numpy as np
import random

# Sets random weights between 0 and 1 for each element in a matrix
def randomWeights(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = random.uniform(0,1)
    return x

# Applying sigmoid function to a value x
def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, inputnodes, n_hiddenLayers, hiddennodes, outputnodes, learningRate):
        self.inputnodes = inputnodes
        self.n_hiddenLayers = n_hiddenLayers
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes

        self.weightsIH = randomWeights(np.zeros((self.inputnodes, self.hiddennodes[0]), dtype=float))
        self.weightsHH = []
        self.biasIHH = []
        for l in range(n_hiddenLayers-1):
            w = randomWeights(np.zeros((self.hiddennodes[l], self.hiddennodes[l+1]), dtype=float))
            self.weightsHH.append(w)
            b = randomWeights(np.zeros((1, self.hiddennodes[l]), dtype=float))
            self.biasIHH.append(b)
        b = randomWeights(np.zeros((1, self.hiddennodes[-1]), dtype=float))
        self.biasIHH.append(b)
        self.weightsHO = randomWeights(np.zeros((self.hiddennodes[-1], self.outputnodes), dtype=float))
        self.biasHO = randomWeights(np.zeros((1, self.outputnodes), dtype=float))
        self.learningRate = learningRate

    def forwardPropagation(self, x):
        # Outputs: Hidden Layer
        a1 = np.dot(self.weights1, x)
        a1 = np.add(a1, self.bias1)
        # Activation function: Hidden Layer
        for num in range(self.hiddennodes):
            a1[0][num] = sigmoid(a1[0][num])

        # Outputs: Output Layer
        a2 = np.dot(self.weights2, a1.T)
        a2 = np.add(a2, self.bias2.T)
        # Activation function: Output Layer
        for num in range(self.outputnodes):
            a2[num][0] = sigmoid(a2[num][0])
        return a2

nn = NeuralNetwork(2,4,[3,4,5,6],3, 0.1)
print("Input -> Hidden")
print(nn.weightsIH)
print("Hidden -> Hidden")
[print(i) for i in nn.weightsHH]
print("Hidden -> Output")
print(nn.weightsHO)
print("Bias Input -> Hidden | Bias Hidden -> Hidden")
[print(i) for i in nn.biasIHH]
print("Bias Hidden -> Output")
print(nn.biasHO)
