from v2 import *
import numpy as np

class Layer( object ):
	"""docstring for Layer"""
	DEFAULT_X_DIMENSION = 1
	DEFAULT_Y_DIMENSION = 1
	SHAPE_HEIGHT_INDEX = 0
	def __init__(self, identity, input_width, input_height, 
		layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
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
		self.rfs = rfs
		self.regular_weights = []
		self.bias_weights = []
		self.regular_weight_changes = []
		self.bias_weight_changes = []
		self.regular_activations = {}
		self.bias_activations = {}
		if ( len( bias_nodes ) == channels ):
			self.bias_nodes = bias_nodes
		else:
			self.bias_nodes = [0] * channels

	def assemble_layer( self ):
		for channel in range( self.channels ):
			# need to cater for non-shared weights; i.e., fcns
			self.regular_weights.append( EdgeGroup( self.regular_weight_init_range, 
				self.layer_width * self.layer_height, self.DEFAULT_Y_DIMENSION ) )
			self.regular_weights[ len( self.regular_weights ) - 1 ].initialise()
			self.regular_weight_changes.append( EdgeGroup( [0.0,0.0], self.layer_width * self.layer_height, self.DEFAULT_Y_DIMENSION ) )
			self.regular_weight_changes[ len( self.regular_weight_changes ) - 1 ].initialise()
			if ( self.biases ):
				self.bias_weights.append( EdgeGroup( self.bias_weight_init_range, 
					self.DEFAULT_X_DIMENSION, self.layer_width * self.layer_height ) )
				self.bias_weights[ len( self.bias_weights ) - 1 ].initialise()
				self.bias_weight_changes.append( EdgeGroup( [0.0,0.0], self.DEFAULT_X_DIMENSION, 
					self.layer_width * self.layer_height ) )
				self.bias_weight_changes[ len( self.bias_weight_changes ) - 1 ].initialise()

	def get_width( self ):
		return self.layer_width 

	def get_channels( self ):
		return self.channels

	def set_channels( self, channels ):
		self.channels = channels

	def get_indices_model( self, channel ):
		if ( channel != None ):
			return np.arange( channel, self.input_width * self.input_height * self.channels, 
				self.channels ).reshape( self.input_height, self.input_width )
		else:
			return np.arange( self.input_height * self.input_width ).reshape( 
				self.input_height, self.input_width)

	def get_height( self ):
		return self.layer_height

	def get_input_width( self ):
		return self.input_width

	def get_input_height( self ):
		return self.input_height

	def get_identity( self ):
		return self.identity

	def get_bias_node( self, channel ):
		return self.bias_nodes[channel]

	def get_regular_weight_init_range( self ):
		return self.regular_weight_init_range

	def get_bias_weight_init_range( self ):
		return self.bias_weight_init_range

	def set_width( self, width ):
		self.layer_width = width

	def set_bias_node( self, node, channel ):
		assert( type( node ) == float )
		self.bias_nodes[ channel ] = node

	def set_rfs( self, rfs ):
		self.rfs = rfs

	def set_height( self, height ):
		self.layer_height = height

	def set_input_width( self, width ):
		self.input_width = width

	def set_input_height( self, height ):
		self.input_height = height

	def set_identity( self, identity ):
		self.identity = identity

	def set_regular_weight_init_range( self, weight_range ):
		assert( type( weight_range ) == list )
		assert( len( weight_range ) == 2 )
		self.regular_weight_init_range = weight_range

	def set_bias_weight_init_range( self, weight_range ):
		assert( type( weight_range ) == list )
		assert( len( weight_range ) == 2 )
		self.bias_weight_init_range = weight_range

	def set_regular_activations( self, activations, channel ):
		assert( np.shape( activations )[self.SHAPE_HEIGHT_INDEX] == self.layer_width * self.layer_height )
		self.regular_activations[channel] = activations

	def get_regular_activations( self, channel = None ):
		if ( channel != None ):
			return self.regular_activations[ channel ]
		return self.regular_activations


	def get_bias_activations( self, channel ):
		return self.bias_activations[ channel ]

	def get_regular_weights( self, channel ):
		return self.regular_weights[ channel ].get_edges()

	def get_bias_weights( self, channel ):
		return self.bias_weights[ channel ].get_edges()

	def set_regular_weights( self, weights, channel ):
		self.regular_weights[ channel ].set_edges( weights )

	def set_bias_weights( self, weights, channel ):
		self.bias_weights[ channel ].set_edges( weights )

	def get_rfs( self ):
		return self.rfs

	def get_regular_weight_changes( self, channel ):
		return self.regular_weight_changes[ channel ].get_edges()

	def get_bias_weight_changes( self, channel ):
		return self.bias_weight_changes[ channel ].get_edges()

	def set_regular_weight_changes( self, changes, channel ):
		self.regular_weight_changes[channel].set_edges( changes )

	def set_bias_weight_changes( self, changes, channel ):
		self.bias_weight_changes[channel].set_edges( changes )



class InputLayer( Layer ):
	"""docstring for InputLayer"""
	def __init__( self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range = None, bias_weight_init_range = None, 
		input_width = None, input_height = None, biases = False ):
		super(InputLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, biases )

	def get_indices_model( self, channel ):
		if ( channel != None ):
			return np.arange( channel, self.layer_width * self.layer_height * self.channels, 
				self.channels ).reshape( self.layer_height, self.layer_width )
		else:
			return np.arange( self.layer_width * self.height ).reshape( 
				self.layer_height, self.layer_width )
		

class ConvolutionalLayer( Layer ):
	"""docstring for ConvolutionalLayer"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(ConvolutionalLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, biases )



class PoolingLayer( Layer ):
	"""docstring for Pooling"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(PoolingLayer, self ).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, biases )

		
class MinPoolingLayer(PoolingLayer):
	"""docstring for MinPooling"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MinPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )
		

class MaxPoolingLayer(PoolingLayer):
	"""docstring for MaxPooling"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MaxPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )
		
class MeanPoolingLayer(PoolingLayer):
	"""docstring for MeanPooling"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(MeanPoolingLayer, self ).__init__( identity, layer_width, 
			layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, input_width, input_height, biases )
"""
def __init__(self, identity, input_width, input_height, 
		layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
		bias_weight_init_range = None, biases = True )
"""


class FullyConnectedLayer( Layer ):
	def __init__(self, identity,
		layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
		bias_weight_init_range,  input_width, input_height, biases = True ):
		super( FullyConnectedLayer, self ).__init__( identity, input_width, input_height, 
			layer_width, layer_height, channels, rfs, bias_nodes, 
			regular_weight_init_range, bias_weight_init_range )

	def assemble_layer( self ):
		print( "input_width: " + str( self.input_width ) )
		print( "input_height: " + str( self.input_height ) )
		print( "layer_width: " + str( self.layer_width ) )
		print( "layer_height: " + str( self.layer_height ) )
		print( "FCN-specific assemble() invoked" )
		for channel in range( self.channels ):
			self.regular_weights.append( EdgeGroup( self.regular_weight_init_range, 
				self.layer_width * self.layer_height * self.input_width * self.input_height, 
				self.DEFAULT_Y_DIMENSION ) )
			self.regular_weight_changes.append( EdgeGroup( [0.0,0.0], 
				self.layer_width * self.layer_height * self.input_width * self.input_height, 
				self.DEFAULT_Y_DIMENSION ) )
			self.regular_weights[ len( self.regular_weights ) - 1 ].initialise()
			self.regular_weight_changes[ len( self.regular_weight_changes )  - 1 ].initialise()
			if ( self.biases ):
				self.bias_weights.append( EdgeGroup( self.bias_weight_init_range, 
					self.DEFAULT_X_DIMENSION, self.layer_width * self.layer_height ) )
				self.bias_weights[ len( self.bias_weights ) - 1 ].initialise()
				self.bias_weight_changes.append( EdgeGroup( [0,0], 
					self.DEFAULT_X_DIMENSION, self.layer_width * self.layer_height ) )
				self.bias_weight_changes[ len( self.bias_weight_changes ) - 1 ].initialise()
		print( "regular weights within assemble(): " + str( self.regular_weights[0].get_edges() ) )
		print( "bias weights within assemble(): " + str( self.bias_weights[0].get_edges() ) ) 


class OutputLayer(Layer):
	"""docstring for OutputLayer"""
	def __init__(self, identity, layer_width, layer_height, channels, rfs, bias_nodes,
		regular_weight_init_range, bias_weight_init_range, 
		input_width, input_height, biases = True ):
		super(OutputLayer, self).__init__( identity, input_width, 
			input_height, layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
			bias_weight_init_range, biases )

		
