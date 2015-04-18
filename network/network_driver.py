from collections import namedtuple
from math import exp
from os.path import realpath
import sys

import numpy as np
from pyspark import SparkContext
from ArtificialNeuralNetwork import *
from validate_layer_config import *
import logging


# inefficient
"""
def parseDataPoint(line):
	parsed_point_features = np.array([float(x) for x in line.split(',')])
	print "parsed_point_features: " + str(parsed_point_features)
	return LabeledPoint(parsed_point_features[0], parsed_point_features[1:])
"""

def readPointBatch(iterator):
	strs = list(iterator)
	print("strs: " + str(strs))
	if (np.size(strs) != 0):
		matrix = np.zeros((len(strs), NUM_INPUT_FEATURES+1))
		for i in xrange(len(strs)):
			matrix[i] = np.append(np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')[NUM_OUTPUT_FEATURES:], 1.0)
		return [matrix]
	return []

NUM_INPUT_FEATURES = 2
NUM_OUTPUT_FEATURES = 1
EMPTY_PARTITION = 0
num_examples = 2

def readOutputPointBatch(iterator):
	strs = list(iterator)
	if (np.size(strs) != 0):
		matrix = np.zeros((len(strs), NUM_OUTPUT_FEATURES))
		for i in xrange(len(strs)):
			matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')[:NUM_OUTPUT_FEATURES]
		return [matrix]
	return []

def checkWeightGradients(network, input_data, target_output):
	epsilon = 10**(-4)
	weights_before_forward_propagation_l0 = network.get_layers().get_layer(0).get_weights()
	weights_before_forward_propagation_l1 = network.get_layers().get_layer(network.get_layers().size()-1).get_weights()
	params_addition = np.zeros((len(weights_before_forward_propagation_l1), 1))
	params_subtraction = np.zeros((len(weights_before_forward_propagation_l1), 1))
	rdd = network.feed_forward(input_data)
	network.back_propagate(rdd, target_output)

	for i in range(len(weights_before_forward_propagation_l1)):
		params_addition[i] = 1.0
		params_subtraction[i] = 1.0
		gradient_addition = np.add(weights_before_forward_propagation_l1, epsilon * params_addition)
		gradient_subtraction = np.subtract(weights_before_forward_propagation_l1, epsilon * params_subtraction)
		network.get_layers().get_layer(0).set_weights(weights_before_forward_propagation_l0)
		network.get_layers().get_layer(1).set_weights(gradient_addition)
		rdd = network.feed_forward(input_data)
		gradient_approx_comp_1 = squared_error(rdd, target_output)
		#gradient_approx_comp_1 = cost_function(rdd, target_output, num_examples)
		network.get_layers().get_layer(1).set_weights(gradient_subtraction)
		rdd = network.feed_forward(input_data)
		gradient_approx_comp_2 = squared_error(rdd, target_output)
		#gradient_approx_comp_2 = cost_function(rdd, target_output, num_examples)
		gradient_approximation = gradient_approx_comp_1.zip(gradient_approx_comp_2)
		print("approximation: " + str(np.mean(np.subtract(np.concatenate((gradient_approximation.collect()[0][0], 
			gradient_approximation.collect()[1][0]), axis = 0), np.concatenate((gradient_approximation.collect()[0][1], 
			gradient_approximation.collect()[1][1]), axis = 0))/(2.0 * epsilon), axis = 0)))


if __name__ == "__main__":
	logging.basicConfig(filename = 'nn.log', level=logging.INFO)
	logging.debug('This message should appear on the console.')
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: distributednn <file> <iterations>"

	print "<file>: " + str(sys.argv[1])
	sc = SparkContext(appName="DistributedNN.py")
	lines = sc.textFile(sys.argv[1], minPartitions = 6)
	data = lines.mapPartitions(readPointBatch).cache()
	print("data: " + str(data.collect()))
	for partition in data.collect():
		if (np.size(partition) == EMPTY_PARTITION):
			print("SYSTEM EXIT DUE TO EMPTY PARTITION")
			sys.exit(1)

	# print("parall 2: " + str(sc.parallelize([0,1,2,3,4,5,6,7,8], 3).coalesce(2).glom().collect()))
	summarised = data.map(lambda data_input: np.mean(data_input)).persist()
	target_output = lines.mapPartitions(readOutputPointBatch).cache()
	nn = ArtificialNeuralNetwork()

	#checkWeightGradients(nn, data, target_output)

	proposed_configurations = [InputLayerType(NUM_INPUT_FEATURES, 3.0, 
		10**(-4), [-1.0,+1.0]), 
		HiddenFCNLayerType(4, 5.0, 10**(-4), [-1.0,+1.0]),
		OutputFCNLayerType(NUM_OUTPUT_FEATURES)]
	nn.build(proposed_configurations)
	layer_index = 0
	for layer in nn.get_layers().get_layers():
		print("Layer index: " + str(layer_index))
		print("Learning rate: " + str(nn.get_layers().get_layer(layer_index).get_learning_rate()))
		print("Weight penalty: " + str(nn.get_layers().get_layer(layer_index).get_weight_penalty()))
		layer_index += 1
	print("num. layers: " + str(nn.get_layers().size()))
	nn.train(data, target_output, int(sys.argv[2]))
	print("size of layer 0: " + str(nn.get_layers().get_layer(0).get_depth()))
	print("end")
	sc.stop()
