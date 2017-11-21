"""
Utility functions for ann module.
"""
import numpy as np


def sigmoid(z):
    """Calculate activation of nodes in a layer.

    Parameters
    ----------
    z : np.ndarray
        vector of inputs

    Returns
    -------
    a : np.ndarray
        sigmoid activations
    cache : np.ndarray
        weighted inputs for caching purpose
    """

    a = 1 / (1 + np.exp(-z))
    cache = z
    return a, cache


def relu(z):
    """

    z: np.ndarray
        vector of inputs

    Returns
    -------
    a : np.ndarray
        RELU activations
    cache : np.ndarray
        weighted inputs for caching purpose
    """
    a = np.maximum(0, z)
    cache = z
    return a, cache


def relu_backward(dA, cache):
    """
    Calculate gradient of the cost with respect to relu activation function.

    Parameters
    ----------
    dA: np.ndarray
        Gradient of the cost function with respect to activations
    cache: np.ndarray
        Cached value of Z

    Returns
    -------
    dZ: np.ndarray
        Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Calculate gradient of the cost with respect to sigmoid activation function.

    Parameters
    ----------
    dA: np.ndarray
        Gradient of the cost function with respect to activations
    cache: np.ndarray
        Cached value of Z

    Returns
    -------
    dZ: np.ndarray
        Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)

    return dZ
