# viewer.py
from __future__ import print_function
import Pyro4

class User(object):
	def __init__(self):
		pass


def main():
	#viewer = Viewer()
	#daemon = Pyro4.Daemon()
	#daemon.register(viewer)
	network = Pyro4.Proxy("PYRONAME:example.neuralnetwork")
	network.build_network(
		[InputLayerType(NUM_INPUT_FEATURES, 3.0, 
		10**(-4), [-1.0,+1.0]), 
		HiddenFCNLayerType(18, 5.0, 10**(-4), [-1.0,+1.0]),
		OutputFCNLayerType(NUM_OUTPUT_FEATURES)])
	driver.train(1, 4)
	driver.get_spark_context().stop()

if __name__ == "__main__":
	main()
