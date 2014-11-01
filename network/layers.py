class Layer(object):
	"""docstring for Layer"""
	def __init__(self, identity, input_width, input_height, 
		layer_width, layer_height, regular_weight_init_range, 
		bias_weight_init_range = None, biases = True ):
		self.identity = identity 
		self.input_width = input_width
		self.input_height = input_height 
		self.layer_width = layer_width 
		self.layer_height = layer_height
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range
		self.biases = biases

	def assemble_layer( self ):
		pass

	def get_width( self ):
		return self.layer_width 

	def get_height( self ):
		return self.layer_height

	def get_input_width( self ):
		return self.input_width

	def get_input_height( self ):
		return self.input_height

	def get_identity( self ):
		return self.identity

	def get_regular_weight_init_range( self ):
		return self.regular_weight_init_range

	def get_bias_weight_init_range( self ):
		return self.bias_weight_init_range



class InputLayer( Layer ):
	"""docstring for InputLayer"""
	def __init__( self, identity, layer_width, layer_height, 
		regular_weight_init_range = None, bias_weight_init_range = None, 
		input_width = None, input_height = None, biases = False ):
		super(InputLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, regular_weight_init_range, 
			bias_weight_init_range, biases )
		

class ConvolutionalLayer( Layer ):
	"""docstring for ConvolutionalLayer"""
	def __init__(self, identity, layer_width, layer_height,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(ConvolutionalLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, regular_weight_init_range, 
			bias_weight_init_range, biases )


class PoolingLayer( Layer ):
	"""docstring for Pooling"""
	def __init__(self, identity, layer_width, layer_height, 
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(PoolingLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, regular_weight_init_range, 
			bias_weight_init_range, biases )

		
class MinPoolingLayer(PoolingLayer):
	"""docstring for MinPooling"""
	def __init__(self, identity, layer_width, layer_height, 
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MinPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, input_width, input_height, regular_weight_init_range, 
			bias_weight_init_range, biases )
		

class MaxPoolingLayer(PoolingLayer):
	"""docstring for MaxPooling"""
	def __init__(self, identity, layer_width, layer_height, 
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MaxPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, input_width, input_height, regular_weight_init_range, 
			bias_weight_init_range, biases )
		
class MeanPoolingLayer(PoolingLayer):
	"""docstring for MeanPooling"""
	def __init__(self, identity, layer_width, layer_height, 
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MeanPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, input_width, input_height, regular_weight_init_range, 
			bias_weight_init_range, biases )

class OutputLayer(Layer):
	"""docstring for OutputLayer"""
	def __init__(self, identity, layer_width, layer_height,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(OutputLayer, self).__init__( identity, input_width, 
			input_height, layer_width, layer_height, regular_weight_init_range, 
			bias_weight_init_range, biases )

class FullyConnectedLayer(Layer):
	"""docstring for FullyConnectedLayer"""
	def __init__(self, identity, layer_width, layer_height,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(FullyConnectedLayer, self).__init__( identity, input_width, 
			input_height, layer_width, layer_height, regular_weight_init_range, 
			bias_weight_init_range, biases )
		