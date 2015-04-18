# -*- coding: utf-8 -*-
from math import * 
import numpy as np
from os.path import realpath
import sys
from pyspark import SparkContext


def readPointBatch(iterator):
	strs = list(iterator)
	print("len(strs): " + str(len(strs)))
	matrix = np.zeros((len(strs), NUM_INPUT_FEATURES+1))
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


# naive cost function implementation - should work for single-output
# networks; i.e., binary classification. -- needs to be refactored for 
# multi-variable binary classification. 
def cost_function(hypothesis, target, num_examples):
	hypothesis_target = hypothesis.zip(target)
	test = hypothesis_target.map(lambda components: 
		components[0])
	print("test.collect(): " + str(test.collect()))
	#return hypothesis_target.map(lambda components: 
	#	(-1.0/num_examples) * (((1.0 - components[1]) * np.log(
	#		1.0 - components[0])) + 
	#		(components[1] * np.log(components[0]))))

	return hypothesis_target.map(lambda components: 
		-components[1] * np.log(components[0]) - (1-components[1]) * np.log(1 - components[0]))

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print >> sys.stderr, "Usage: distributednn <file> <iterations>"

	sc = SparkContext(appName = "Test.py")
	lines = sc.textFile(sys.argv[1])
	data = lines.mapPartitions(readPointBatch).cache()
	print("data.collect(): " + str(data.collect()))
	transformation = data.map(lambda x: x * 2)
	print("transformation.collect(): " + str(transformation.collect()))
	combined = data.zip(transformation)
	output = combined.map(lambda components: 
		components[0] + components[1])
	print("output.collect(): " + str(output.collect()))

	






