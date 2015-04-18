from pyspark import SparkContext

if __name__ == "__main__":
	sc = SparkContext()
	nums = sc.parallelize([1,2,3,4])
	squared = nums.map(lambda x: x * x).collect()
	for num in squared:
		print "%i" % (num)