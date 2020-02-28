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
    assert(H.shape == Y.shape)
    assert(H.shape == (8, 1))
    minval=0.0000000001
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
    len_prev = len(prev)
    len_to = len(to_layer)
    for i in range(len_to):
        # to_layer[i] = prev.dot(weight[:,i:i+1])
        to_layer[i] = (prev * weight[:, i:i+1]).sum(axis = 0)
    for i in range(len_to):
        to_layer[i] += bias[0, i]
        to_layer[i] = sigmoid(to_layer[i])

def error(Y, Z):
    len_Y = len(Y)
    err = (Z - Y) * (Z) * (1 - Z)
    return err


def train(input_X, Y, W0, W1, B0, B1, lmb, learning_rate = 0.01, iter_num = 100):
    global OUTPUT_HISTORY

    history = [None] * iter_num
    cost_history = [None] * iter_num
    triW0 = np.zeros(W0.shape)
    triW1 = np.zeros(W1.shape)
    triB0 = np.zeros((1, 3))
    triB1 = np.zeros((1, 8))
    hidden_layer = np.zeros((3, 1))
    output_layer = np.zeros((8, 1), dtype = np.float128)

    for i in range(iter_num):
        assert(hidden_layer.shape == (3, 1))
        assert(output_layer.shape == (8, 1))
        forward_propgt(input_X, W0, B0, hidden_layer)
        forward_propgt(hidden_layer, W1, B1, output_layer)

        history[i] = np.append(output_layer, i)
        ccost = cost_function(output_layer, Y)
        cost_history[i] = ccost
        # propagate back
        # ...
        err2 = error(Y, output_layer)
        assert(err2.shape == (8, 1))
        dw1 = hidden_layer.dot(err2.T)
        assert(dw1.shape == (3,8))
        db1 = err2.T
        assert(db1.shape == (1, 8))

        err1 = np.zeros((3, 1))
        temp = W1.dot(err2)
        # w_sum = temp.sum(axis = 1)
        err1 = (1 - hidden_layer) * hidden_layer * (temp)
        assert(err1.shape == (3, 1))

        dw0 = input_X * err1.T
        db0 = err1.T
        assert(db0.shape == (1, 3))

        triW0 += dw0
        triB0 += db0
        triW1 += dw1
        triB1 += db1
        # Update W0
        W0 = W0 - learning_rate * ((1 / HIDDEN_NODES) * triW0 + lmb * W0)
        B0 = B0 - learning_rate * ((1 / 3) * triB0)
        W1 = W1 - learning_rate * ((1 / OUTPUT_NODES) * triW1 + lmb * W1)
        B1 = B1 - learning_rate * ((1 / 8) * triB1)
    forward_propgt(input_X, W0, B0, hidden_layer)
    forward_propgt(hidden_layer, W1, B1, output_layer)
    history.append(output_layer)
    final_biases = [B0, B1]
    final_weights = [W0, W1]
    return history, cost_history, final_weights, final_biases

def draw_cost(cost_history, several):
    plt.title("Cost")
    plt.ylabel("Cost")
    plt.xlabel("Iteration(s)")
    plt.grid(True)
    xlim = 500
    if several == True:
        plt.xlim(0, xlim)
        plt.ylim(0, 1)
        plt.plot(np.arange(0, len(cost_history)), np.array(cost_history), color = 'black')
        return
    rates = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    colors = ['magenta', 'blue', 'green', 'red', 'cyan', 'black']
    for i in range(len(rates)):
        _, costs, _, _ = train(input_x, input_x, first_layer_weights, hidden_layer_weights, b0, b1, lmb, rates[i], ITERATION_NUM)
        len_his = len(costs)
        plt.xlim(0, xlim)
        plt.ylim(0, 1)
        plt.plot(np.arange(0, len_his), np.array(costs), label = 'learning rate: {0:.3f}'.format(rates[i]), color = colors[i])
        plt.legend()


def init():
    plt.xlim(0, 1)
    plt.ylim(0, 8)
    ln.set_data(dX[0], dY)
    return ln,

if __name__ == "__main__":
    inputs = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype = np.float128)
    for i in range(7):
        inputs = np.vstack((inputs, [1 if _-1 == i else 0 for _ in range(8)]))
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

    fig = plt.figure(1)
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

    draw_cost(cost_his, False)
    plt.savefig("cost.png")
    dX = history
    dY = [1, 2, 3, 4, 5, 6, 7, 8]
    fig2 = plt.figure(2)
    # plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    ln, = plt.plot([], [], "ro")
    # mywriter = animation.FFMpegWriter(fps = 60)
    # ani.save('convrg.mp4',writer=mywriter)

    plt.show()
