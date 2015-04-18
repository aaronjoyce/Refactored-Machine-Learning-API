import numpy as np 
from pyspark import SparkContext

if __name__ == "__main__":
	sc = SparkContext("DistributedNN.py")
	pairs = sc.parallelize([1, 2, 3, 4, 2, 4, 1]).map(lambda x: (x, x))
	#print("pairs.collect(): " + str(pairs.collect()))