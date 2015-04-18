from collections import namedtuple
from math import exp
from os.path import realpath
import sys

import numpy as np
from pyspark import SparkContext
from ArtificialNeuralNetwork import LabeledPoint

# inefficient
"""
def parseDataPoint(line):
	parsed_point_features = np.array([float(x) for x in line.split(',')])
	print "parsed_point_features: " + str(parsed_point_features)
	return LabeledPoint(parsed_point_features[0], parsed_point_features[1:])
"""

def readPointBatch(iterator):
	print("Executed")
	strs = list(iterator)
	print("type(strs): " + str(np.size(strs)))
	if (np.size(strs) != 0):
		matrix = np.zeros((len(strs), D+1))
		for i in xrange(len(strs)):
			matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')
		return [matrix]
	else:
		return []
D = 2

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: distributednn <file> <iterations>"

	print "<file>: " + str(sys.argv[1])
	sc = SparkContext(appName="DistributedNN.py")
	lines = sc.textFile(sys.argv[1], 7)
	print("lines: " + str(lines.collect()))
	data = lines.mapPartitions(readPointBatch).cache()
	print("data: " + str(data.collect()))
	suitable_partitioning_found = False

	"""
	while not(suitable_partitioning_found):
		problem_found = False
		for (partition)
	"""
	"""
	w = 2 * np.random.ranf(size=D) - 1

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

	iterations = int(sys.argv[2])
	for i in range(iterations):
		print "On iteration %i" % (i+1)
		w -= data.map(lambda m: gradient(m, w)).reduce(add)
	"""
	sc.stop()
