import numpy as np

def random_matrix(x, y, init_range):
	return np.array(np.random.uniform(init_range[0], init_range[1], 
		x * y).reshape(x, y))

class Layer(object):
	def __init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases = True):
		self.input_depth = input_depth
		self.output_depth = output_depth
		self.init_range = init_range
		self.weights = random_matrix(input_depth+1 if with_biases else input_depth, 
			output_depth, 
			self.init_range)
		self.weight_penalty = weight_penalty
		self.learning_rate = learning_rate
		self.with_biases = with_biases

	# @returns a matrix of weights that represent 
	# the Layer instance. If 'with_biases' is true, 
	# the matrix of weights returned include weights
	# on edges connecting the bias activation of layer l-1
	# to the all non-bias activations of layer l. 
	# Form matrix form, see forms_dev.py
	def get_weights(self):
		return self.weights

	# @param weights: A matrix of real-value numbers, that must be of 
	# the same shape as the existing layer's weights. 
	# @throws: an AssertionError if the 'weights' argument
	# does not have the correct form. This exception should 
	# remain un-checked, as it is useful to identify explicit architectural
	# errors. 
	def set_weights(self, weights):
		assert np.shape(weights) == np.shape(self.weights)
		self.weights = weights

	# @returns a real-value number that represents the learning 
	# rate/factor associated with the Layer instance. This is used 
	# during back-propagation through the network of which it forms part. 
	def get_learning_rate(self):
		return self.learning_rate

	# @param weights: A real-value number that represents
	# the learning rate/factor associated with the Layer instance. 
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	# @returns a real-value number that represents 
	# the weight penalty associated with the Layer instance. 
	# The weight penalty is factor of the degree of adjustment
	# applied to the Layer instance's weights, in order to satisfy
	# some mathematical objective function.  
	def get_weight_penalty(self):
		return self.weight_penalty

	# @param weights: 
	def set_weight_penalty(self, weight_penalty):
		self.weight_penalty = weight_penalty

	def is_biased(self):
		return self.with_biases

	def get_depth(self):
		return self.input_depth

	def get_init_range(self):
		return self.init_range

	def get_higher_connected_layer_depth(self):
		return self.output_depth

class InputLayer(Layer):
	"""docstring for InputLayer"""
	def __init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases = True):
		Layer.__init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases)
		

class OutputLayer(Layer):
	"""docstring for OutputLayer"""
	def __init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases = True):
		Layer.__init__(self, input_depth, utput_depth, 
		init_range, weight_penalty, learning_rate, with_biases)
		pass

class FullyConnectedLayer(Layer):
	"""docstring for FullyConnectedLayer"""
	def __init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases = True):
		Layer.__init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases)
		pass


