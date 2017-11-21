"""This program re-implements the neural network created in the assignment.
"""
import numpy as np
from ann_utils import *


class Network(object):
    """Class for implementing basic shallow neural networks with 1 hidden layer
    for binary classification of input data consisting of N-dimensional vectors.

    Attributes
    ----------
    n_x : int
        Number of nodes in input layer
    n_h : int
        Number of nodes in the hidden layer
    n_y : int
        Number of nodes in output layer

    """

    def __init__(self, layer_dims):
        """Initialize the network.

        Parameters
        ----------
        layer_dims : list
            List consisting of number of nodes in each layer.
        """
        self.layer_dims = layer_dims
        self.caches = []
        self.parameters = self.initialize_params()

    def initialize_params(self, scaling_factor=0.01):
        """Initialize parameters of the network.

        Parameters
        ----------
        scaling_factor : float
            A scaling factor for the random initialization of weight matrices. Default=0.01

        Returns
        -------
        params : dict
            A dictionary of network parameters

        """
        params = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            params["W" + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * scaling_factor
            params["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert params["W" + str(l)].shape == (self.layer_dims[l], self.layer_dims[l - 1]), \
                "Mismatch in weight matrix dimenions of layer %d\n" % l
            assert params["b" + str(l)].shape == (self.layer_dims[l], 1), \
                "Mismatch in bias vector dimenions of layer %d\n" % l

        return params

    @staticmethod
    def weighted_inputs(A, W, b):
        """Calculates weight inputs for a layer l from activations from the
        previous layer using parameters of the current layer.

        Parameters
        ----------
        A : np.ndarray
            Activations from previous layer
        W : np.ndarray
            Weight matrix of current layer
        b: np.ndarray
            Bias of current layer

        Returns
        -------
        Z : np.ndarray
            Weighted inputs for nodes in the layer
        cache : tuple (A, W, b)
            A tuple consisting of activations in the previous layer (A),
            current weight matrix (W) of the layer, biases (b) of the layer.
        """

        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1])), "Incorrect dimenions of activation vector."
        cache = (A, W, b)

        return Z, cache

    def activations(self, A_prev, W, b, non_linearity):
        """Calculates activations of the current layer.

        Parameters
        ----------
        A_prev : np.ndarray
            Activations of the previous layer
        W : np.ndarray
            Weight matrix of current layer
        b : np.ndarray
            bias of the current layer
        non_linearity : str
            Type of activation

        Returns
        -------
        A : np.ndarray
            Output activations
        cache : tuple
            Tuple that stores current A, W, b and z for caching purposes
        """

        z, cache_weighted_inputs = self.weighted_inputs(A_prev, W, b)
        if non_linearity == "sigmoid":
            a, cache_activation = sigmoid(z)
        elif non_linearity == "relu":
            a, cache_activation = relu(z)
        else:
            raise KeyError("Unrecognized activation function='{0}'".format(non_linearity))

        assert (a.shape == (W.shape[0], A_prev.shape[1]))
        cache = (cache_weighted_inputs, cache_activation)

        return a, cache

    def forward_propagation(self, X):
        """Forward propagation for an L-layer model. This only supports a model with L - 1
        layers with RELU activations and a final layer with  sigmoid activation.

        Parameters
        ----------
        X : np.ndarray
            input data matrix

        Returns
        -------
        AL : np.ndarray
            Activations of the last layer, which make the output of the network

        caches : list
            A list of consisting of caches corresponding to each layer
        """

        caches = []
        A = X
        L = len(self.parameters.keys()) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.activations(A_prev, self.parameters["W"+str(l)], self.parameters["b"+str(l)], "relu")
            caches.append(cache)

        AL, cache = self.activations(A, self.parameters["W"+str(L)], self.parameters["b"+str(L)], "sigmoid")
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Calculates cost of the network output against target values based on cross-entropy cost function.

        Parameters
        ----------
        AL : np.ndarray
            Network output
        Y : np.ndarray
            Target values

        Returns
        -------
        cost : float
            Value of the cost function
        """

        m = Y.shape[1]

        cost = -np.sum((Y * np.log(AL)) + ((1 - Y) * (np.log(1 - AL)))) / m
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return cost

    def linear_backward(self, dZ, cache):
        """
        Calculate the linera portion of backward propagation.

        Parameters
        ----------
        dZ : np.ndarray
            Gradient of the cost with respect to weight inputs
        cache : tuple
            Cached values of activations of layer l-1 and weight matrix and bias of layer l.

        Returns
        -------
        dW : np.ndarray
            Gradient of the cost with respect to weights of the current layer
        db : np.ndarray
            Gradient of the cost with respect to bias of the current layer
        dA_prev : np.ndarray
            Gradient of cost with respect to activations of the previous layer
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Calculate gradient of the cost function with respect to weight input

        Parameters
        ----------
        dA : np.ndarary
            Gradient of activations
        cache : tuples
            Stored values of A, W, b and Z for the current layer.
        activation : str
            Type of activation function

        Returns
        -------
        dA_prev : np.ndarray
            Gradient of cost with respect to activations of the previous layer
        dW : np.ndarray
            Gradient of the cost with respect to weights of the current layer
        db : np.ndarray
            Gradient of the cost with respect to bias of the current layer
        """

        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, W, b = self.linear_backward(dZ, linear_cache)
        if activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, W, b = self.linear_backward(dZ, linear_cache)

        return dA_prev, W, b

    def backward_propagation(self, AL, Y, caches):
        """
        Implements backprop for a an L-layer model. This only supports a model with L - 1
        layers with RELU activations and a final layer with  sigmoid activation.

        Parameters
        ----------
        AL : np.ndarray
            Activations of the final layer
        Y : np.ndarray
            Vector/Matrix of target values
        caches : list
            A list of caches, (A, W, b) and Z values of the layer

        Returns
        -------
        grads : dict
            A dictionary of layer-wise gradients
        """
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        # get dA[L]
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        # Iterate over hidden layers
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+ str(l + 2)], current_cache, "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, params, grads, eta):
        """
        Implements gradient descent update of parameters
        
        Parameters
        ----------
        params : dict
            A dictionary of network parameters
        grads : dict
            A dictionary of layer-wise gradients

        eta : float
            Learning rate

        Returns
        -------
        params : dict
            Updated parameters        
        """

        L = len(params) // 2 # number of layers in the neural network
        for layer in range(1, L+1):
            params["W" + str(layer)] = params["W" + str(layer)] - (eta * grads["dW" + str(layer)])
            params["b" + str(layer)] = params["b" + str(layer)] - (eta * grads["db" + str(layer)])
 
        return params