import numpy as np 
from pyspark import SparkContext
from pyspark import RDD
from random import * 

if __name__ == "__main__":
	sc = SparkContext(appName = "ArtificialNetworkLayers")
	a_rdd = RDD(None, sc)