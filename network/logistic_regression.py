from collections import namedtuple
from math import exp
from os.path import realpath
import sys

import numpy as np
from pyspark import SparkContext

D = 4

def readPointBatch(iterator):
	strs = list(iterator)
	print("len(strs): " + str(len(strs)))
	matrix = np.zeros((len(strs), D+1))
	for i in xrange(len(strs)):
		print("strs[i]: " + str(strs[i]))
		print("strs[i].replace(',', ' '): " + str(strs[i].replace(',', ' ')))
		print("np.shape(matrix[i]): " + str(np.shape(matrix[i])))
		print("np.fromstring(): " + str(
			np.shape( np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' '))))
		matrix[i] = np.fromstring(strs[i].replace(',', ' '), dtype = np.float32, sep = ' ')
	return [matrix]

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: logistic_regression <file> <iterations>"
		exit(-1)

	sc  = SparkContext(appName = "PythonLR", batchSize = 5)
	# the function you pass mapPartitions must take an iterable of your RDD type and
	# return an iterable of the same or some other type. 
	points = sc.textFile(sys.argv[1]).mapPartitions(readPointBatch).cache()
	iterations = int(sys.argv[2])

	# Initialise w to a random value
	w = 2 * np.random.ranf(size=D) - 1
	print "Initial w: " + str(w)

	# Compute logistic regression gradient for a matrix of data points
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

	for i in range(iterations):
		print "On iteration %i" % (i + 1)
		w -= points.map(lambda m: gradient(m, w)).reduce(add)

	print "Final w: " + str(w)

	sc.stop()
