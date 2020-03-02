import csv
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt

LAYERS = 3
INPUT_NODES = 12
OUTPUT_NODES = 2
HIDDEN_NODES = 6
ALPHA = 0.01
OUTPUT_HISTORY = []

def sigmoid(x):
    y = 1 / (1 + np.exp(x * -1))
    return y

def cost_function(H, Y):
    m = len(Y)
    minval = -np.inf
    try:
        for i in range(len(H)):
            if H[i] == 0: H[i] = minval
            if (1 - H[i] == 0): H[i] = 1 - minval
        _sum = np.sum(Y * np.log(H) + (1 - Y) * (np.log(1 - H)))
    except ZeroDivisionErro:
        pass
    cost = (-1 / m) * _sum
    return cost

def forward_propgt(prev, weight, bias, to_layer):
    len_to = len(to_layer)
    for i in range(len_to):
        # to_layer[i] = prev.dot(weight[:,i:i+1])
        to_layer[i] = (prev * weight[:, i:i+1]).sum(axis = 0)
    for i in range(len_to):
        to_layer[i] += bias[0, i]
        to_layer[i] = sigmoid(to_layer[i])

def error(Y, Z):
    err = (Z - Y) * (Z) * (1 - Z)
    return err


def train(input_X, Y, W0, W1, B0, B1, lmb, learning_rate = 0.01, iter_num = 100):
    global OUTPUT_HISTORY

    history = [None] * iter_num
    cost_history = [None] * iter_num
    triW0 = np.zeros(W0.shape)
    triW1 = np.zeros(W1.shape)
    triB0 = np.zeros((1, HIDDEN_NODES))
    triB1 = np.zeros((1, OUTPUT_NODES))
    hidden_layer = np.zeros((HIDDEN_NODES, 1))
    output_layer = np.zeros((OUTPUT_NODES, 1), dtype = np.float128)

    for i in range(iter_num):
        forward_propgt(input_X, W0, B0, hidden_layer)
        forward_propgt(hidden_layer, W1, B1, output_layer)

        history[i] = np.append(output_layer, i)
        ccost = cost_function(output_layer, Y)
        cost_history[i] = ccost
        # propagate back
        # ...
        err2 = error(Y, output_layer)
        dw1 = hidden_layer.dot(err2.T)
        db1 = err2.T

        err1 = np.zeros((HIDDEN_NODES, 1))
        temp = W1.dot(err2)
        # w_sum = temp.sum(axis = 1)
        err1 = (1 - hidden_layer) * hidden_layer * (temp)

        dw0 = input_X * err1.T
        db0 = err1.T

        triW0 += dw0
        triB0 += db0
        triW1 += dw1
        triB1 += db1
        # Update W0
        W0 = W0 - learning_rate * ((1 / HIDDEN_NODES) * triW0 + lmb * W0)
        B0 = B0 - learning_rate * ((1 / HIDDEN_NODES) * triB0)
        W1 = W1 - learning_rate * ((1 / OUTPUT_NODES) * triW1 + lmb * W1)
        B1 = B1 - learning_rate * ((1 / OUTPUT_NODES) * triB1)
    forward_propgt(input_X, W0, B0, hidden_layer)
    forward_propgt(hidden_layer, W1, B1, output_layer)
    history.append(output_layer)
    final_biases = [B0, B1]
    final_weights = [W0, W1]
    return history, cost_history, final_weights, final_biases



if __name__ == "__main__":
    inputs = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype = np.float128)
    for i in range(INPUT_NODES - 1):
        inputs = np.vstack((inputs, [1 if _-1 == i else 0 for _ in range(INPUT_NODES)]))
    first_layer_weights_random = np.random.rand(INPUT_NODES, HIDDEN_NODES)
    first_layer_weights_zero = np.zeros((INPUT_NODES, HIDDEN_NODES))
    hidden_layer_weights_zero = np.zeros((HIDDEN_NODES, OUTPUT_NODES))
    hidden_layer_weights_random = np.random.rand(HIDDEN_NODES, OUTPUT_NODES)
    first_layer_weights = first_layer_weights_random
    hidden_layer_weights = hidden_layer_weights_random
    b0 = np.ones((1, HIDDEN_NODES))
    b1 = np.ones((1, OUTPUT_NODES))
    print("Start Training")
    lmb = 0.15
    ALPHA = 0.01
    ITERATION_NUM = 5000
    input_x = inputs[0].reshape(INPUT_NODES, 1)
    history, cost_his, weights, biases = train(input_x, input_x, first_layer_weights, hidden_layer_weights, b0, b1, lmb, ALPHA, ITERATION_NUM)
    # def train(input_X, Y, W0, W1, B0, B1, lmb, learning_rate = 0.01, iter_num = 100):

    # fig = plt.figure(1)
    print("Weights of the 1st(intput) layer")
    print(weights[0])
    print("Weights of the hidden layer")
    print(weights[1])
    print("Bias node in the 1st(input) layer")
    print(biases[0])
    print("Bias node in the hidden(input) layer")
    print(biases[1])
    print("The networks take the input:")
    print(input_x)
    print("After {} iteration, the outputs are:".format(ITERATION_NUM))
    print(history[-1][:-1])
