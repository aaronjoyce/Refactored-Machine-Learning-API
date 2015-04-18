from pyspark import SparkContext
from random import * 
import numpy as np

if __name__ == "__main__":
	def sample(p):
		x, y = random(), random()
		print("value returned: " + str(1 if x * x + y * y < 1 else 0))
		return 1 if x * x + y * y < 1 else 0 

	spark = SparkContext(appName = "Test.py")

	seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
	combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
	data = np.array([[20,19,18,17,16],[15,14,13,12,11],[10,9,8,7,6],[5,4,3,2,1]])
	print("data.shape: " + str(data.shape))
	parallelized = spark.parallelize(data).aggregate((np.zeros((1,data.shape[1])),0), seqOp, combOp)
	print("parallelized: " + str(parallelized[0]/parallelized[1]))

	test = spark.parallelize(np.zeros((4,1)))

	print("test: " + str(test.collect()))