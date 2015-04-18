from __future__ import print_function
from pyspark import SparkContext 
from ArtificialNeuralNetwork import *
import sys
import Pyro4
from validate_layer_config import *
import numpy as np
from layer import *
from network_driver import *

if sys.version_info < (3, 0):
	input = raw_input

class NeuralNetworkAPI(object):
	def __init__(self, spark_context):
		self.spark_context = spark_context
		self.nn = ArtificialNeuralNetwork() # will be used as part of implementation

	# @param layer_configs: List of Python dicts, where
	# the following hyper-parameters of each layer may 
	# be specified: 
	#	'num_nodes': 			<a positive integer value>
	#	'learning_rate':		<a non-zero real value>
	#	'weight_penalty':		<a non-zero real value>
	#	'weight_init_range':	<a two-element list>
	# E.g., layer_configs = [{'num_nodes' : 5, 'learning_rate' : 3.0, 
	#	'weight_penalty' : 0.0001, 'weight_init_range' : [-1.0,+1.0]}, {'num_nodes' : 8, 'learning_rate' : 3.0, 
	#	'weight_penalty' : 0.0001, 'weight_init_range' : [-1.0,+1.0]}, {'num_nodes' : 2, 'learning_rate' : 3.0, 
	#	'weight_penalty' : 0.0001, 'weight_init_range' : [-1.0,+1.0]}], 
	# where the above network configuration denotes a three-layer neural network, 
	# featuring five input nodes, 8 hidden nodes, and two output nodes.
	# Important: The order in which the Python dicts are specified determines the order
	# of the corresponding network's layers. Hence, the first dict defines layer 0, the 
	# second dict defines layer 1, etc. 
	def build(self, layer_configs):
		print("is invoked")
		highest_layer_index = len(layer_configs)
		proposed_configurations = []
		for layer_index, layer_configuration in enumerate(layer_configs):
			if (layer_index == 0):
				proposed_configurations.append(InputLayerType(layer_configuration['num_nodes'], 
					layer_configuration['learning_rate'], 
					layer_configuration['weight_penalty'], 
					layer_configuration['weight_init_range']))
			elif (layer_index == (highest_layer_index-1)):
				proposed_configurations.append(OutputFCNLayerType(layer_configuration['num_nodes']))
			else:
				proposed_configurations.append(HiddenFCNLayerType(layer_configuration['num_nodes'], 
					layer_configuration['learning_rate'], 
					layer_configuration['weight_penalty'], 
					layer_configuration['weight_init_range']))
		self.nn.build(proposed_configurations)
		return 1;


	# @param data: List of Python tuples, where each tuple 
	# denotes a training instance. Each training instance is a tuple
	# consisting of two python lists, where the first list of 
	# the tuple contains the "input" parameters, and the second
	# list contains the values the network should optimise for. 
	# @param num_iterations: A positive integer value denoting
	# the number of iterations to be performed during the training process. 
	# @param batch_size: A positive integer value denoting
	# the size of each batch. A batch represents the number of input instance
	# that are grouped in order to improve efficiency during training. 
	# Generally, the large the batch size value, the longer it takes
	# for the network's objective function to converge. 
	def train(self, parameters, targets, num_iterations, batch_size):
		assert(len(parameters) == len(targets))
		num_param_instances = len(parameters)
		parameters = self.spark_context.parallelize(parameters, num_param_instances/batch_size).glom()
		targets = self.spark_context.parallelize(targets, num_param_instances/batch_size).glom()
		self.nn.train(parameters, targets, num_iterations)
		return "self.train(...) invoked"

	# @param inputs: 
	def predict(self, inputs):
		inputs = self.spark_context.parallelize(np.asarray([inputs]))
		predictions = self.nn.feed_forward(inputs).collect()
		for index, prediction in enumerate(predictions):
			predictions[index] = prediction.tolist()
		return predictions

	def add_layer(self, layer_config):
		output_depth = self.nn.get_layers().get_layer(self.nn.num_layers()-1).get_higher_connected_layer_depth()
		self.nn.get_layers().set_layer((self.nn.num_layers()-1), FullyConnectedLayer(self.nn.get_layers().get_layer(self.nn.num_layers()-1).get_depth(), 
			layer_config['num_nodes'], self.nn.get_layers().get_layer(self.nn.num_layers()-1).get_init_range(), 
			self.nn.get_layers().get_layer(self.nn.num_layers()-1).get_weight_penalty(), 
			self.nn.get_layers().get_layer(self.nn.num_layers()-1).get_learning_rate()))
		self.nn.add_layer(FullyConnectedLayer(layer_config['num_nodes'],
			output_depth,
			layer_config['weight_init_range'], layer_config['weight_penalty'], 
			layer_config['learning_rate']))
		return "self.add_layer(...) invoked"


