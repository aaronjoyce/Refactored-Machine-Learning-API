from __future__ import print_function
import Pyro4
from collections import namedtuple
from math import exp
from os.path import realpath
import sys
from threading import Thread
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

class RequestAcceptor:
	def __init__(self):
		self.sc = SparkContext(appName = "GeneralPurposeCluster.py")


	def access_network(self, user_id):
		sc = self.sc

		def provide_access():
			print("method executed")
			driver = None
			def driver_instatiation():
				print("driver_instatiation")
				driver = NetworkDriver(sys.argv[1], user_id, sc)
				print("post driver assignment")
				driver.start()
				print("post start")
			driver_thread = Thread(target = driver_instatiation)
			driver_thread.setDaemon(True)
			driver_thread.start()
			print("after driver instantiation")
			daemon = Pyro4.Daemon()
			driver_uri = daemon.register(driver)
			ns = Pyro4.locateNS()
			print("provide_access()")
			ns.register("examples.networkdriver." + user_id, driver_uri)
			daemon.requestLoop()
		print("Executed")
		thread = Thread(target = provide_access)
		thread.setDaemon(True)
		thread.start()


class NetworkDriver(Thread):
	def __init__(self, filename, user_id, spark_context):
		self.sc = spark_context
		sc_reference = self.sc
		#self.nn = ArtificialNeuralNetwork(sc_reference)
		self.nn = None
		self.filename = filename
		self.user_id = user_id

	def run(self):
		self.nn = ArtificialNeuralNetwork(sc_reference)
		daemon = Pyro4.Daemon()
		nn = self.nn
		network_uri = daemon.register(nn)
		ns = Pyro4.locateNS()
		ns.register("example.neuralnetwork" + self.user_id, network_uri)
		daemon.requestLoop()


	def build_network(self, proposed_configurations):
		self.nn.build(proposed_configurations)

	def train(self, num_iterations, num_partitions):
		lines = self.sc.textFile(self.filename, num_partitions)
		parameters = lines.mapPartitions(readPointBatch).cache()
		targets = lines.mapPartitions(readOutputPointBatch).cache()
		self.nn.train(parameters, targets, num_iterations)

	# @param parameters: Tuple of size one or greater. 
	def predict(self, parameters):
		rdd = self.sc.parallelize(parameters).cache()
		return (self.nn.feed_forward(rdd)).collect()

	def get_spark_context(self):
		return self.sc 

	def get_layers(self):
		return self.nn.get_layers()



def main():
	request_acceptor = RequestAcceptor()
	daemon = Pyro4.Daemon()
	acceptor_uri = daemon.register(request_acceptor)
	ns = Pyro4.locateNS()
	ns.register("example.request_acceptor", acceptor_uri)
	daemon.requestLoop()


if __name__ == "__main__":
	logging.basicConfig(filename = 'nn.log', level=logging.INFO)
	logging.debug('This message should appear on the console.')
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: distributednn <file> <iterations>"
	main()