from collections import namedtuple
from math import exp
from os.path import realpath
import sys

import numpy as np
from pyspark import SparkContext
from ArtificialNeuralNetwork import *
from validate_layer_config import *


# inefficient
"""
def parseDataPoint(line):
	parsed_point_features = np.array([float(x) for x in line.split(',')])
	print "parsed_point_features: " + str(parsed_point_features)
	return LabeledPoint(parsed_point_features[0], parsed_point_features[1:])
"""

def readPointBatch(iterator):
	strs = list(iterator)
	print("len(strs): " + str(len(strs)))
	matrix = np.zeros((len(strs), NUM_INPUT_FEATURES+NUM_OUTPUT_FEATURES))
	for i in xrange(len(strs)):
		matrix[i] = np.append(np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')[NUM_OUTPUT_FEATURES:], 1.0)
	return [matrix]
NUM_INPUT_FEATURES = 4
NUM_OUTPUT_FEATURES = 1

def readOutputPointBatch(iterator):
	strs = list(iterator)
	matrix = np.zeros((len(strs), NUM_OUTPUT_FEATURES))
	for i in xrange(len(strs)):
		matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')[:NUM_OUTPUT_FEATURES]
	return [matrix]


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: distributednn <file> <iterations>"

	print "<file>: " + str(sys.argv[1])
	sc = SparkContext(appName="DistributedNN.py")
	lines = sc.textFile(sys.argv[1])
	data = lines.mapPartitions(readPointBatch).cache()
	w = 2 * np.random.ranf(size=NUM_INPUT_FEATURES) - 1 
	target_output = lines.mapPartitions(readOutputPointBatch).cache()
	nn = ArtificialNeuralNetwork()

	proposed_configurations = [InputLayerType(4), HiddenFCNLayerType(4),
		HiddenFCNLayerType(3), HiddenFCNLayerType(6), OutputFCNLayerType(1)]
	layers = generate_layers(proposed_configurations)
	for layer in layers:
		nn.add_layer(layer)
	#nn.add_layer(Layer(4, 4, [0.1,0.2], 0.1, 3.0))	# Input layer
	#nn.add_layer(Layer(4, 4, [0.1,0.2], 0.3, 3.0))	# Output layer
	#nn.add_layer(Layer(4, 1, [0.1,0.2], 0.3, 3.0, False))

	iterations = int(sys.argv[2])
	
	for i in range(iterations):
		print "iteration: " + str(i)
		rdd = nn.feed_forward(data)
		print "rdd.collect(): " + str(rdd.collect())
		print("target_output: " + str(target_output.collect()))
		nn.back_propagate(rdd, target_output)
	


	print("nn: " + str(nn));
	print("nn.layers: " + str(nn.layers))
	for layer in nn.layers.get_layers():
		print "layer.get_weights(): " + str(layer.get_weights())

	def gradient(matrix, w):
		print("matrix: " + str(matrix))
		print("w: " + str(w))
		Y = matrix[:, 0]	# point labels (first column of input file)
		X = matrix[:, 1:]	# point coordinates
		# For each point (x, y), compute gradient function, then sum these up
		return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

	def add(x, y):
		x += y
		return x
	"""
	iterations = int(sys.argv[2])
	for i in range(iterations):
		print "On iteration %i" % (i+1)
		w -= data.map(lambda m: gradient(m, w)).reduce(add)
	"""

	sc.stop()
