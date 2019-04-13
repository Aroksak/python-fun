#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:00:41 2019

@author: sakar
"""

import numpy as np
import matplotlib.pyplot as plt


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward.
    """
    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError


class Identity(Activation):
    """
    Identity function
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return np.ones_like(self.state)


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))
        self._argument = x
        return self.state

    def derivative(self):
        return (1 - self.state) * self.state


class Tanh(Activation):
    """
    Hyperbolic tangent non-linearity
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - self.state ** 2


class Criterion(object):
    """
    Generic criterion defenition class
    """
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError


class MSE(Criterion):
    """
    Standart loss function: sum of squarred errors
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, a, y):
        self.logits = a
        self.labels = y
        self.loss = np.sum((a-y)**2, axis=1) / 2

    def derivative(self):
        return self.logits - self.labels


class CrossEntropy(Criterion):
    """
    Cross-entropy loss function
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, a, y):
        self.logits = a
        self.labels = y
#        self.loss = -(y*np.log(a) + (1-y) * np.log(1-a)).sum()

    def delta(self):
        return self.logits - self.labels


class Layer(object):
    """
    Class for all layers of neural net
    """
    def __init__(self, rng, n_in, n_out,
                 act_func=Sigmoid(),
                 W=None, b=None):
        """
        Typical layer of a neural network with all neurons interconnected,
        default activation function is sigmoid. Weigth matrix has size
        (n_in, n_out), bias matrix (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of neurons in previous layer

        :type n_out: int
        :param n_out: number of neurons in the current layer

        :type act_func: function(float)
        :param act_func: activation function for each neuron in the layer

        :type criterion: function(logits, labels) or None
        :param criterion: loss-function for MLP training in case of the output
                          layer or None elsewhere.
        """
        if W is None:
            W = rng.uniform(
                low=-6/np.sqrt(n_in + n_out),
                high=6/np.sqrt(n_in + n_out),
                size=(n_in, n_out)
            )

        if b is None:
            b = np.zeros((1, n_out))

        self.W = W
        self.b = b
        self.size = n_out
        self.act_func = act_func
        self.input = None
        self.nextLayer = None

    def forward(self, input):
        self.input = input
        lin_output = input @ self.W + self.b
        self.output = self.act_func(lin_output)
        return self.output

    def derivative(self):
        return self.act_func.derivative()


class MLP(object):
    """
    Class for MultiLayer Perceptron creation.
    Typical MLP, all connections between layers are present. Activation
    function for each layer can be set independently.

    :type rng: numpy.numpy.RandomState
    :param rng: a random number generator used to initialize weights

    :type structure: list or tuple
    :param structure: first element is size of input, then sizes of hidden
        layers and output size.

    :type act_func: Activation function(x) or list of functions
    :param act func: List is given must be of size equal to len(structure)-1,
        and each function of act_func will be will be set for each
        corresponding layer.

    :type criterion: Criterion function
    :param criterion: Loss-function to be used for MLP training.
    """
    def __init__(self, rng,
                 structure,
                 act_func,
                 criterion=MSE()):
        self.hiddenLayer = [Layer(rng=rng,
                                  n_in=structure[i],
                                  n_out=structure[i+1],
                                  act_func=act_func[i])
                            for i in range(len(structure)-2)]
        self.outputLayer = Layer(rng=rng,
                                 n_in=structure[-2],
                                 n_out=structure[-1],
                                 act_func=act_func[-1])
        for i in range(len(self.hiddenLayer) - 1):
            self.hiddenLayer[i].nextLayer = self.hiddenLayer[i+1]
        self.hiddenLayer[-1].nextLayer = self.outputLayer
        self.criterion = criterion
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        for i in range(len(self.hiddenLayer)):
            if i == 0:
                self.hiddenLayer[i].forward(self.input)
            else:
                self.hiddenLayer[i].forward(self.hiddenLayer[i-1].output)
        if len(self.hiddenLayer) > 0:
            self.output = self.outputLayer.forward(self.hiddenLayer[-1].output)
        else:
            self.output = self.outputLayer.forward(self.input)
        return self.output

    def backward(self, y, etta):
        self.criterion(self.output, y)

        if isinstance(self.criterion, CrossEntropy) and \
                isinstance(self.outputLayer.act_func, Sigmoid):
            delta = self.criterion.delta()
        else:
            delta = self.criterion.derivative() * self.outputLayer.derivative()
        self.outputLayer.b -= etta*delta
        self.outputLayer.W -= (self.outputLayer.input.T @ delta) * etta

        for layer in self.hiddenLayer[::-1]:
            delta = (delta @ layer.nextLayer.W.T) *\
                    layer.derivative()
            layer.b -= etta*delta
            layer.W -= (layer.input.T @ delta) * etta

    def train(self, x_train, y_train,
              max_epochs=1000, etta=0.1, auto_stop=True, plot=False):
        epochs_without_improvement = 0
        prev_eval = np.inf
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.ion()
            fig.show()
            fig.canvas.draw()
            evs = []
        for n in range(max_epochs):
            for i in range(len(x_train)):
                self.forward(x_train[i])
                self.backward(y_train[i], etta)
            current_eval = 1 - np.equal(
                    np.argmax(self.transform(x_train), axis=2),
                    np.argmax(y_train, axis=2)
                    ).sum() / x_train.shape[0]
            if plot:
                ax.clear()
                evs.append(current_eval)
                ax.plot(evs)
                fig.canvas.draw()
            if auto_stop:
                if current_eval < prev_eval:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement > 10:
                        return
                prev_eval = current_eval

    def transform(self, x):
        result = np.empty((x.shape[0], 1, self.outputLayer.size))
        for i in range(x.shape[0]):
            result[i] = self.forward(x[i])
        return result


if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    appr = MLP(rng, [1, 2, 2, 1], [Sigmoid(), Sigmoid(), Identity()])
    x = rng.uniform(
            low=0.0,
            high=1.0,
            size=(1000, 1, 1)
            )
    y = np.log(1+x)
    x_train = x[:900]
    y_train = y[:900]
    x_test = x[900:]
    y_test = y[900:]
    appr.train(x_train, y_train, 100, auto_stop=False)
