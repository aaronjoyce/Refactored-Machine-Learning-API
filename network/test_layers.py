from layer import *

if __name__ == "__main__":

	input_layer = InputLayer(5, 5, [0.1, 0.2], 0.1, 3.0)
	fcn_layer = FullyConnectedLayer(5, 5, [0.1, 0.2], 0.1, 3.0)
	output_layer = OutputLayer(5, 5, [0.1, 0.2], 0.1, 3.0)

	print("type(input_layer): " + str(isinstance(input_layer, InputLayer)))