from v2 import *
import numpy as np

class Layer( object ):
	"""docstring for Layer"""
	DEFAULT_X_DIMENSION = 1
	DEFAULT_Y_DIMENSION = 1
	def __init__(self, identity, input_width, input_height, 
		layer_width, layer_height, channels, regular_weight_init_range, 
		bias_weight_init_range = None, biases = True ):
		self.identity = identity 
		self.input_width = input_width
		self.input_height = input_height 
		self.layer_width = layer_width 
		self.layer_height = layer_height
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range
		self.biases = biases
		self.channels = channels
		self.regular_weights = []
		self.bias_weights = []
		self.regular_activations = []
		self.bias_activations = []

	def assemble_layer( self ):
		print( "assemble_layer invoked" )
		print( "self.channels: " + str( self.channels ) )
		for channel in range( self.channels ):
			self.regular_weights.append( EdgeGroup( self.regular_weight_init_range, 
				self.layer_width * self.layer_height, self.DEFAULT_Y_DIMENSION ) )
			self.regular_weights[ len( self.regular_weights ) - 1 ].initialise()
			if ( self.biases ):
				self.bias_weights.append( EdgeGroup( self.bias_weight_init_range, 
					self.DEFAULT_X_DIMENSION, self.DEFAULT_Y_DIMENSION ) )

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

	def set_width( self, width ):
		self.layer_width = width

	def set_height( self, height ):
		self.layer_height = height

	def set_input_width( self, width ):
		self.input_width = width

	def set_input_height( self, height ):
		self.input_height = height

	def set_identity( self, identity ):
		self.identity = identity

	def set_regular_weight_init_range( self, weight_range ):
		self.regular_weight_init_range = weight_range

	def set_bias_weight_init_range( self, weight_range ):
		self.bias_weight_init_range = weight_range

	def set_regular_activations( self, activations, channel ):
		self.regular_activations = activations

	def set_bias_activations( self, activations, channel ):
		self.bias_activations = activations

	def get_regular_activations( self, channel ):
		return self.regular_activations[ channel ]

	def get_bias_activations( self, channel ):
		return self.bias_activations[ channel ]

	def get_regular_weights( self, channel ):
		print( "channel: " + str( channel ) )
		return self.regular_weights[ channel ]

	def get_bias_weights( self, channel ):
		return self.bias_weights[ channel ]

	def set_regular_weights( self, weights, channel ):
		self.regular_weights[ channel ] = weights

	def set_bias_weights( self, weights, channel ):
		self.bias_weights[ channel ] = weights





class InputLayer( Layer ):
	"""docstring for InputLayer"""
	def __init__( self, identity, layer_width, layer_height, channels,
		regular_weight_init_range = None, bias_weight_init_range = None, 
		input_width = None, input_height = None, biases = False ):
		super(InputLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, biases )
		

class ConvolutionalLayer( Layer ):
	"""docstring for ConvolutionalLayer"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(ConvolutionalLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, biases )


class PoolingLayer( Layer ):
	"""docstring for Pooling"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(PoolingLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, biases )

		
class MinPoolingLayer(PoolingLayer):
	"""docstring for MinPooling"""
	def __init__(self, identity, layer_width, layer_height, channels, 
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MinPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )
		

class MaxPoolingLayer(PoolingLayer):
	"""docstring for MaxPooling"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MaxPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )
		
class MeanPoolingLayer(PoolingLayer):
	"""docstring for MeanPooling"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MeanPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )

class OutputLayer(Layer):
	"""docstring for OutputLayer"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(OutputLayer, self).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, biases )

class FullyConnectedLayer(Layer):
	"""docstring for FullyConnectedLayer"""
	def __init__(self, identity, layer_width, layer_height, channels,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(FullyConnectedLayer, self).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, regular_weight_init_range, 
			bias_weight_init_range, biases )
		
