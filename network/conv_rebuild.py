import types
from convolutional_network_types import * 
from network_validation_tools import *
import numpy as np
from layer_constraints import LayerConstraints
from convolutional_network_types import *
from layers import *
from data_formatting import *
from v2 import EdgeGroup

"""
from v2 import Network
from convolutional_layer_2 import ConvolutionalLayer
from pooling import PoolingLayer
from pooling import compute_pooling_layer_depth
from pooling import compute_pooling_layer_breadth
from v2 import get_random_permutation
from v2 import randomise
from v2 import get_batch_order
from mnist_loader import load_data
from mnist_loader import load_data_wrapper
from mnist_loader import vectorized_result
"""

ROW_STEP = 1
ALONG_VERTICAL_AXIS = 0
APPEND_ROW = 0

class ConvolutionalNetwork:
	def __init__( self, layer_constraints, 
		learning_rate, identity ):
		self.layer_constraints = layer_constraints
		self.learning_rate = learning_rate
		self.identity = identity
		self.layers = []

	def assemble_network( self ):
		for layer in self.layer_constraints:
			if ( isinstance( self.layer_constraints[layer].get_super_type(), 
				InputType ) ):
				self.layers.append( InputLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(), self.layer_constraints[layer].get_receptive_field_size() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				ConvolutionalType ) ):
				self.layers.append( ConvolutionalLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				OutputType ) ):
				self.layers.append( OutputLayer( self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(),
					self.layer_constraints[ layer ].get_channels(), 
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				FullyConnectedType ) ):
				self.layers.append( FullyConnectedLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				PoolingType ) ):
				if ( isinstance( self.layer_constraints[layer].get_sub_type(), 
					MinPoolingType ) ):
					self.layers.append( MaxPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(), 
					MeanPoolingType ) ):
					self.layers.append( MeanPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) ) 
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(),
					MaxPoolingType ) ):
					self.layers.append( MaxPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

				else:
					raise Exception( "Invalid pooling type specified as sub-type of PoolingType" ) 
			else:
				raise Exception( "Invalid LayerConstraints object specified" )
			if len( self.layers ) != 1:
				self.layers[ len( self.layers ) - 1 ].assemble_layer()

	def hypothesis( self, inputs ):
		for layer in range( len( self.layers ) ):
			if layer == 1: 
				computed_activations = [np.empty(( self.layers[layer].get_height() * self.layers[layer].get_width(),
					np.shape( inputs )[1] ))] * self.layers[layer].get_channels()
				for channel in range( self.layers[layer].get_channels() ):
					indices = self.layers[layer-1].get_indices_model(channel)
					for row, col in self.iterate_over_input_groups( 0, 
						self.layers[layer].get_input_width(), 
						self.layers[layer].get_input_height(), self.layers[layer].get_rfs() ):		
						computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )
					self.layers[layer].set_regular_activations( computed_activations[channel], channel )

			elif layer > 1:
				computed_activations = [np.empty(( self.layers[layer].get_height() * self.layers[layer].get_width(),
					np.shape( inputs )[1] ))] * self.layers[layer].get_channels()
				for channel in range( self.layers[layer].get_channels() ):
					indices = self.layers[layer].get_indices_model( channel )
					
					for row, col in self.iterate_over_input_groups( 0, 
						self.layers[layer].get_input_width(), 
						self.layers[layer].get_input_height(), self.layers[layer].get_rfs() ):

						# Must collate regular activations to form 
						# a matrix composed of all input channels, 
						# rather than channel-specific matrices. 
						# Otherwise, write an alternative implementation. 
						computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )
					self.layers[layer].set_regular_activations( computed_activations[channel], channel )
					

		



	def iterate_over_input_groups( self, start, input_width, input_height, rfs ):
		for row in range( input_height - rfs + 1 ):
			for col in range( input_width - rfs + 1 ):
				yield row, col


	def get_layers( self ):
		return self.layers

	# this concerns pre-programming aspects
	# such as randomisation of batches
	def preprocess_inputs( self ):
		pass



if __name__ == "__main__":

	# 'input_layer_width'			 : int
	# 'input_layer_height' 			 : int
	# 'proposed_layer_types_and_rfs' : dict of dicts, where the key of the 
	# primary dict denotes the layer index, and the value is a sub-dict, where
	# the key of the sub-dict is a layer object type, and the corresponding value
	# denotes the proposed receptive field size, passed as an int-type value. 

	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 2 }, 2 : { MaxPoolingType() : 4 }, 
		3 : { FullyConnectedType() : 1 } }
	input_layer_width = 7
	input_layer_height = 7
	instances_per_batch = 4
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 3

	# returns a one-dimensional key-value dict. 
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
		regular_weight_init_range, bias_weight_init_range, channels )
	network = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N." )
	network.assemble_network()

	inputs = np.empty(( 
		input_layer_width * input_layer_height * channels ) * instances_per_batch )
	inputs = inputs.reshape( input_layer_width * input_layer_height * channels, instances_per_batch )

	for i in range( input_layer_width * input_layer_height * channels ):
		inputs[i].fill( i + 1 )
	network.hypothesis( inputs )


	"""	
	for layer in range( len( network.get_layers() ) ):
		print( "layer: " + str( layer ) )
		print( "layer height: " + str( network.get_layers()[layer].get_height() ) )
		print( "layer width: " + str( network.get_layers()[layer].get_width() ) )
		if ( layer != 0 ):
			print( "regular weights: " + str( network.get_layers()[layer].get_regular_weights(0).get_edges() ) )
		print( "\n\n" ) 
	"""




