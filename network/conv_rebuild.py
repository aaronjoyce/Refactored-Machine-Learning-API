import types
from convolutional_network_types import * 
from network_validation_tools import *
import numpy as np
from layer_constraints import LayerConstraints
from convolutional_network_types import *
from layers import *

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



class ConvolutionalNetwork:
	def __init__( self, layer_constraints, 
		learning_rate, identity ):
		self.layer_constraints = layer_constraints
		self.learning_rate = learning_rate
		self.identity = identity
		self.layers = []

	def assemble_network( self ):
		for layer in self.layer_constraints:
			print( "layer: " + str( layer ) )
			if ( isinstance( self.layer_constraints[layer].get_super_type(), 
				InputType ) ):
				self.layers.append( InputLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(), self.layer_constraints[layer] ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				ConvolutionalType ) ):
				self.layers.append( ConvolutionalLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				OutputType ) ):
				self.layers.append( OutputLayer( self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(),
					self.layer_constraints[ layer ].get_channels(), 
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				FullyConnectedType ) ):
				self.layers.append( FullyConnectedLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
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
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(), 
					MeanPoolingType ) ):
					self.layers.append( MeanPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) ) 
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(),
					MaxPoolingType ) ):
					print( "max-pooling.regular_weight_init_range: " + str( 
						self.layer_constraints[layer].get_regular_weight_init_range() ) )
					print( "min-pooling.bias_weight_init_range: " + str( 
						self.layer_constraints[layer].get_bias_weight_init_range() ) )
					self.layers.append( MaxPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
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
		pass

	def get_layers( self ):
		return self.layers

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
		3 : { FullyConnectedType() : 3 } }
	input_layer_width = 7
	input_layer_height = 7
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 3

	# returns a one-dimensional key-value dict. 
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
		regular_weight_init_range, bias_weight_init_range, channels )
	network = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N." )
	network.assemble_network()


	
	for layer in range( len( network.get_layers() ) ):
		print( "layer: " + str( layer ) )
		print( "layer height: " + str( network.get_layers()[layer].get_height() ) )
		print( "layer width: " + str( network.get_layers()[layer].get_width() ) )
		if ( layer != 0 ):
			print( "regular weights: " + str( network.get_layers()[layer].get_regular_weights(0).get_edges() ) )
		print( "\n\n" ) 





