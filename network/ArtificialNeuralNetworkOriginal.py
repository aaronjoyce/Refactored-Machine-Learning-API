import numpy as np 
from pyspark import SparkContext
import logging

def sigmoid( x ):
	return 1/(1+np.exp(-x))

class ArtificialNeuralNetwork:
	BASE_WEIGHT_GRADIENTS_INDEX = 0
	def __init__(self):
		self.layers = Layers()
		self.activations = []
		self.weight_gradients = []

	# @param sc: A SocketContext object 
	# instantiated by the Network Driver, and
	# passed as part of a function invocation upon
	# a ArtificialNeuralNetwork instance. 
	# @returns an RDD that represents the output layer
	# activations. 
	def feed_forward(self, sc):
		rdd = None
		self.activations.append(sc)
		for layer in range(self.num_layers()):
			weights = self.layers.get_layer(layer).get_weights()
			if layer == 0:
				rdd = sc.map(lambda units: 1/(1+np.exp(-np.dot(units, weights)))).persist()
				self.activations.append(rdd)
			else:
				if (self.layers.get_layer(layer).is_biased() == True):
					rdd = rdd.map(lambda units: 1/(1+np.exp(-np.dot(np.append(units, 
						np.ones((len(units), 1)), axis = 1), weights)))).persist()
					self.activations.append(rdd)
				else:
					rdd = rdd.map(lambda units: 1/(1+np.exp(-np.dot(units, weights)))).persist()
					self.activations.append(rdd)
		return rdd

	# @param 
	# @returns ...
	def back_propagate(self, hypothesis, target):
		output_error = None
		hidden_error = None
		weight_gradients = [None] * self.layers.size()
		for layer in range(self.num_layers()-1, -1, -1):
			logging.info('layer: ' + str(layer))
			if (layer == self.num_layers()-1):
				hypothesis_target = hypothesis.zip(target)
				output_error = hypothesis_target.map(lambda part: 
					-(part[1] - part[0])*(part[0]*(1-part[0]))).persist()
				lower_layer_activations = self.activations[layer];
				error_activations = output_error.zip(lower_layer_activations).persist()
				weight_gradients[layer] = error_activations.map(lambda components: np.dot(np.transpose(components[1]),
					components[0])).persist()
				logging.info('errors: ' + str(output_error.collect()))
				for i in range(len(weight_gradients[layer].collect())):
					logging.info('errors shape: ' + str(output_error.collect()[i].shape))
				logging.info('weight gradients: ' + str(weight_gradients[layer].collect()))
				for i in range(len(weight_gradients[layer].collect())):
					logging.info('weight gradients shape: ' + str(weight_gradients[layer].collect()[i].shape))
				for i in range(len(weight_gradients[layer].collect())):
					
					logging.info('weights before: ' + str(self.layers.get_layer(layer).get_weights()))
					logging.info('weights before shape: ' + str(self.layers.get_layer(layer).get_weights().shape))
					self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1] -= (self.layers.get_layer(layer).get_learning_rate() * ((1.0/(len(weight_gradients[layer].collect()[i]))) * weight_gradients[layer].collect()[i] + 
							self.layers.get_layer(layer).get_weight_penalty() * self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1]))
					self.layers.get_layer(layer).get_weights()[len(self.layers.get_layer(layer).get_weights())-1] -= (self.layers.get_layer(layer).get_learning_rate() * ((1.0/(len(weight_gradients[layer].collect()[i]))) * np.mean(output_error.collect()[i], axis = 0)))
					logging.info('weights after: ' + str(self.layers.get_layer(layer).get_weights()))
					logging.info('weights after shape: ' + str(self.layers.get_layer(layer).get_weights().shape))
			else:
				error_activations = None
				weights = self.layers.get_layer(layer+1).get_weights()
				if (layer == self.num_layers()-2):
					layer_activations = self.activations[layer+1]
					weights = self.layers.get_layer(layer+1).get_weights()
					error_activations = output_error.zip(layer_activations).persist()
					hidden_error = error_activations.map(lambda part: np.multiply(np.dot(part[0], 
						np.transpose(weights[:len(weights)-1])), np.multiply(part[1], (1-part[1])))).persist()
					layer_activations = self.activations[layer]
					error_activations = hidden_error.zip(layer_activations).persist()

					if (weight_gradients[layer] == None):
						weight_gradients[layer] = error_activations.map(lambda components: np.dot(np.transpose(components[1]), 
							components[0])).persist()
				else:
					layer_activations = self.activations[layer+1]
					error_activations = hidden_error.zip(layer_activations).persist()
					hidden_error = error_activations.map(lambda part: np.dot(part[0], 
						np.transpose(weights[:len(weights)-1]))).persist()
					error_activations = hidden_error.zip(self.activations[layer]).persist()
					if (layer != 0):
						if (weight_gradients[layer] == None):
							weight_gradients[layer] = error_activations.map(lambda components: np.dot(np.transpose(components[1]), 
								components[0])).persist()
						else:
							running_gradients = error_activations.map(lambda components: np.dot(np.transpose(components[1]), components[0])).persist()
							new_gradients_existing_gradients = weight_gradients[layer+1].zip(running_gradients).persist()
							weight_gradients[layer] = new_gradients_existing_gradients.map(lambda components: components[0] + components[1]).persist()
					else:
						if (weight_gradients[layer] == None):
							weight_gradients[layer] = error_activations.map(lambda components: np.dot(np.transpose(components[1][:,:components[1].shape[1]-1]),
								components[0])).persist()
						else:
							running_gradients = error_activations.map(lambda components: np.dot(np.transpose(components[1][:,:components[1].shape[1]-1]), components[0])).persist()
							new_gradients_existing_gradients = weight_gradients[layer+1].zip(running_gradients).persist()
							weight_gradients[layer] = new_gradients_existing_gradients.map(lambda components: components[0] + components[1]).persist()
				logging.info('errors: ' + str(hidden_error.collect()))
				for  i in range(len(hidden_error.collect())):
					logging.info('errors shape: ' + str(hidden_error.collect()[i].shape))
				logging.info('weights gradients: ' + str(weight_gradients[layer].collect()))
				for i in range(len(weight_gradients[layer].collect())):
					logging.info('weight gradients shape: ' + str(weight_gradients[layer].collect()[i].shape))
				for i in range(len(weight_gradients[layer].collect())):
					logging.info('weights before: ' + str(self.layers.get_layer(layer).get_weights()))
					logging.info('weights before shape: ' + str(self.layers.get_layer(layer).get_weights().shape))
					self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1] -= (self.layers.get_layer(layer).get_learning_rate() * ((1.0/(len(weight_gradients[layer].collect()[i]))) * weight_gradients[layer].collect()[i] + 
							self.layers.get_layer(layer).get_weight_penalty() * self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1]))
					self.layers.get_layer(layer).get_weights()[len(self.layers.get_layer(layer).get_weights())-1] -= self.layers.get_layer(layer).get_learning_rate() * ((1.0/(len(weight_gradients[layer].collect()[i]))) * np.mean(hidden_error.collect()[i], axis = 0))
					logging.info('weights after: ' + str(self.layers.get_layer(layer).get_weights()))
					logging.info('weights after shape: ' + str(self.layers.get_layer(layer).get_weights().shape))

		for i in weight_gradients:
			if i != None:
				print("i.collect(): " + str(i.collect()))

		self.activations = []		

	def add_layer(self, layer):
		self.layers.add(layer)

	# @returns the number of layer that 
	# comprise the multi-layer perceptron. 
	# type: int. 
	def num_layers(self):
		return len(self.layers.get_layers())

class LabeledPoint:
	def __init__(self, label, features):
		self.label = label 
		if type(features) == np.ndarray:
			self.features = features 
		else:
			raise TypeError("Type of features in no NumPy ndarray.")

	def __str__(self):
		return "(" + ",".join((str(self.label), str(self.features))) + ")"

class Layers:
	def __init__(self):
		self.layers = []

	def add(self, layer):
		self.layers.append(layer)

	def size(self):
		return len(self.layers)

	def pop(self, index):
		self.layers.pop([index])

	# @return an array of Layer objects. 
	def get_layers(self):
		return self.layers
	# 
	def get_layer(self, index):
		assert(index < len(self.layers))
		return self.layers[index]

# Assumption: A fully-connected layer 
# is desired. 
# Interface: Layer(layer_depth, next_layer_depth, 
# 	init_range, with_biases = True)
# @param layer_depth:
# @param next_layer_depth:
# @param init_range:
# @param weight_penalty:
# @param learning_rate:
# @param with_biases 

def random_matrix(x, y, init_range):
	return np.array(np.random.uniform(init_range[0], init_range[1], 
		x * y).reshape(x, y))

def add(x, y):
		x += y
		return x

class Layer:
	def __init__(self, input_depth, output_depth, 
		init_range, weight_penalty, learning_rate, with_biases = True):
		self.input_depth = input_depth
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
