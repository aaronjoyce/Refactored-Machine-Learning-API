# -*- coding: utf-8 -*-
from __future__ import print_function
import Pyro4
import numpy as np 
from pyspark import SparkContext
import logging
from naive_cost_function import cost_function
from validate_layer_config import generate_layers
from layer import *

def sigmoid(x):
	return 1/(1+np.exp(-x))

def squared_error(x, y):
	hypothesis_target = x.zip(y)
	return hypothesis_target.map(lambda components: 0.5 * (abs(components[0] - components[1])**2.0))

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
		print("sc.collect(): " + str(sc.collect()))
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

	def build(self, layer_configurations):
		layers = generate_layers(layer_configurations)
		for layer in layers:
			self.add_layer(layer)
	
	def train(self, inputs, targets, iterations):
		print("inputs (train): " + str(train.collect()))
		for iteration in range(iterations):
			hypothesis = self.feed_forward(inputs)
			self.back_propagate(hypothesis, targets)
	
	# @param 
	# @returns ...
	def back_propagate(self, hypothesis, target):
		output_error = None
		hidden_error = None
		weight_gradients = [None] * self.layers.size()
		temp = None
		for layer in range(self.num_layers()-1, -1, -1):
			if (is_adjacent_to_output_layer(layer, self.num_layers())):
				hypothesis_target = hypothesis.zip(target).persist()
				# Original cost function:
				output_error = hypothesis_target.map(lambda part: 
					-(part[1] - part[0])*(part[0]*(1-part[0]))).persist()
				output_error = output_error.map(lambda x: np.mean(x, axis = 0))
				# Softmax cost function:
				#output_error = cost_function(hypothesis, target, 2) # - will need to concatenate first.
				out_error = output_error
				lower_layer_activations = self.activations[layer]
				lower_layer_activations = lower_layer_activations.map(lambda x: np.mean(x, axis = 0)).persist()
				error_activations = output_error.zip(lower_layer_activations).persist()
				weight_gradients[layer] = error_activations.map(lambda x: np.transpose(np.dot(np.transpose(np.matrix(x[0])), np.matrix(x[1])))).persist()
				#weight_gradients[layer] = error_activations.map(lambda components: np.transpose(np.dot(np.transpose(np.matrix(components[1])),
				#	components[0]))).persist()
				seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
				combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
				output_error = output_error.aggregate((np.zeros((output_error.collect()[0].shape[0],len(output_error.collect()[0]))),0), seqOp, combOp)
				output_error = output_error[0]/float(output_error[1])
				seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
				combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
				weight_gradients[layer] = weight_gradients[layer].aggregate((np.zeros((weight_gradients[layer].collect()[0].shape[0],weight_gradients[layer].collect()[0].shape[1])),0), seqOp, combOp)
				weight_gradients[layer] = weight_gradients[layer][0]/float(weight_gradients[layer][1])
			else:
				error_activations = None
				weights = self.layers.get_layer(layer+1).get_weights()
				# Optimisation to be made here if this is the second-highest layer - doesn't need 
				# to re-compute activations. 
				current_layer_activations = self.activations[layer+1].map(lambda x: 
					np.mean(x, axis = 0)).persist()
				if (layer == self.num_layers()-2):
					h_error = out_error
				else:
					h_error = hidden_error
				hidden_error_activations = h_error.zip(current_layer_activations).persist()
				hidden_error = hidden_error_activations.map(lambda components: np.multiply(np.dot(components[0], 
					np.transpose(weights[:len(weights)-1])), np.multiply(components[1], (1-components[1])))).persist()
				temp = hidden_error
				lower_layer_activations = self.activations[layer].map(lambda x: 
					np.mean(x, axis = 0)).persist()
				errors_activations = hidden_error.zip(lower_layer_activations).persist()
				# Calculation of the weight gradient connecting this layer to the 
				# next lower layer. 
				weight_gradients[layer] = errors_activations.map(lambda x: np.transpose(np.dot(np.transpose(np.matrix(x[0])), np.matrix(x[1]))))
				seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
				combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
				hidden_error = hidden_error.aggregate((np.zeros((np.matrix(hidden_error.collect()[0]).shape[0],len(hidden_error.collect()[0]))),0), seqOp, combOp)
				hidden_error = np.transpose(hidden_error[0]/float(hidden_error[1]))
				seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
				combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
				weight_gradients[layer] = weight_gradients[layer].aggregate((np.zeros((weight_gradients[layer].collect()[0].shape[0],weight_gradients[layer].collect()[0].shape[1])),0), seqOp, combOp)
				weight_gradients[layer] = weight_gradients[layer][0]/float(weight_gradients[layer][1])

			# Update 'regular' weights
			self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1] -= (
				self.layers.get_layer(layer).get_learning_rate() * ((weight_gradients[layer] if 
					layer != 0 else weight_gradients[layer][:weight_gradients[layer].shape[0]-1]) + 
						self.layers.get_layer(layer).get_weight_penalty() * \
						self.layers.get_layer(layer).get_weights()[:self.layers.get_layer(layer).get_weights().shape[0]-1]))
			# Update 'bias' weights
			self.layers.get_layer(layer).get_weights()[self.layers.get_layer(layer).get_weights().shape[0]-1] -= (
				self.layers.get_layer(layer).get_learning_rate()) * hidden_error.flatten() if layer != (self.num_layers()-1) else output_error[0]
			if (temp != None):
				hidden_error = temp
		self.activations = []		

	def add_layer(self, layer):
		self.layers.add(layer)

	# @returns the number of layer that 
	# comprise the multi-layer perceptron. 
	# type: int. 
	def num_layers(self):
		return len(self.layers.get_layers())

	def get_layers(self):
		return self.layers


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

	def set_layer(self, index, layer):
		assert(index <len(self.layers))
		selfl.layers[index] = layer 

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

def add(x, y):
		x += y
		return x

def is_adjacent_to_output_layer(layer_index, num_layers):
	return layer_index == (num_layers-1)


if __name__ == "__main__":
	pass
