from layer import *


class LayerType:
	def __init__(self, depth, learning_rate, 
		weight_regularisation, 
		weight_initialisation_range):
		self.depth = depth
		self.learning_rate = learning_rate
		self.weight_regularisation = weight_regularisation
		self.weight_initialisation_range = weight_initialisation_range

	def get_depth(self):
		return self.depth

	def get_learning_rate(self):
		return self.learning_rate

	def get_weight_regularisation(self):
		return self.weight_regularisation

	def get_weight_initialisation_range(self):
		return self.weight_initialisation_range

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def set_weight_regulariation(self, weight_regularisation):
		self.weight_regularisation = weight_regularisation

	def set_weight_initialisation_range(self, weight_initialisation_range):
		self.weight_initialisation_range = weight_initialisation_range

class InputLayerType(LayerType):
	def __init__(self, depth, learning_rate, 
		weight_regularisation, weight_initialisation_range):
		LayerType.__init__(self, depth, learning_rate, 
			weight_regularisation, weight_initialisation_range)
		pass

class HiddenFCNLayerType(LayerType):
	def __init__(self, depth, learning_rate, 
		weight_regularisation, weight_initialisation_range):
		LayerType.__init__(self, depth, learning_rate, 
			weight_regularisation, weight_initialisation_range)
		pass

class OutputFCNLayerType(LayerType):
	def __init__(self, depth):
		LayerType.__init__(self, depth, None, 
			None, None)
		pass

def generate_layers(configurations):
	layers = []
	for i in range(len(configurations)-1):
		if (isinstance(configurations[i], InputLayerType)):
			layers.append(InputLayer(configurations[i].get_depth(), 
				configurations[i+1].get_depth(), configurations[i].get_weight_initialisation_range(), 
				configurations[i].get_weight_regularisation(), 
				configurations[i].get_learning_rate()))
		elif (isinstance(configurations[i], HiddenFCNLayerType)):
			layers.append(FullyConnectedLayer(configurations[i].get_depth(), 
				configurations[i+1].get_depth(), configurations[i].get_weight_initialisation_range(), 
				configurations[i].get_weight_regularisation(), 
				configurations[i].get_learning_rate()))
		elif (isinstance(configurations[i], OutputFCNLayerType)):
			layers.append(OutputLayer(configurations[i].get_depth(), 
				configurations[i+1].get_depth(), configurations[i].get_weight_initialisation_range(), 
				configurations[i].get_weight_regularisation(), 
				configurations[i].get_learning_rate()))

	return layers

if __name__ == "__main__":
	proposed_configurations = [InputLayerType(4), HiddenFCNLayerType(4),
		OutputFCNLayerType(1)]
	layers = generate_layers(proposed_configurations)
	print("layers: " + str(layers))

	
