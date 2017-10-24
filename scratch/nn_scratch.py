"""This program re-implements the neural network created in the assignment.
"""
import numpy as np

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

	def __init__(self, layer_sizes):
		"""Initialize the network.
		
		Parameters
		----------
		layer_sizes : list
			List consisting of number of nodes in each layer.
		"""
		assert len(layer_sizes) == 3, "Currently only one hidden layer supported."
		self.n_x = layer_sizes[0]
		self.n_h = layer_sizes[1]
		self.n_y = layer_sizes[2]
		self.parameters = {"W1": None, "b1": None, "W2": None, "b2": None}

	def initialize_params(self):
		"""Initialize parameters of the network.
		
		Returns
		-------
		params : dict
			Dictionary of parameters associated with the network.
		"""

		W1 = np.random.randn()
		