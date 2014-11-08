# network_validation_tools.py
import types
from convolutional_network_types import * 
from network_validation_tools import *
import numpy as np
from layer_constraints import LayerConstraints
from convolutional_network_types import *

from build_network import compute_convolutional_layer_depth
from build_network import compute_convolutional_layer_breadth
from build_network import is_dimensionality_compatible
from build_network import is_odd
from build_network import universal_is_dimensionality_compatible


def generate_layer_configurations( input_layer_width, 
	input_layer_height, proposed_layer_types_and_rfs, 
	regular_weight_init_range, bias_weight_init_range, channels ):

	rfr = []
	layer_configurations = {}
	for primary_key in proposed_layer_types_and_rfs:
		for sub_key in proposed_layer_types_and_rfs[ primary_key ]:
			if ( proposed_layer_types_and_rfs[ primary_key ][ sub_key ] != 0 ):
				rfr.append( proposed_layer_types_and_rfs[primary_key][sub_key] )

	if ( universal_is_dimensionality_compatible( input_layer_width, rfr, 1 ) ):
		for p_key in range( len( proposed_layer_types_and_rfs ) ):
			if ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], InputType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
						input_layer_width, input_layer_height, s_key, channels, regular_weight_init_range, 
						bias_weight_init_range, InputType() )
			
			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], OutputType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints(
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, OutputType(), FullyConnectedType() )	
	

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], FullyConnectedType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_width(),
						proposed_layer_types_and_rfs[ p_key ][ s_key], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1) ].get_height(),
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, FullyConnectedType() )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], ConvolutionalType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, ConvolutionalType() )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MinPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_width(),
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_height(),
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MinPoolingType() ) 
				
			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MaxPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MaxPoolingType() )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MeanPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations[ 'layer ' + str( p_key ) ] = LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ 'layer ' + str( p_key -1 ) ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ 'layer ' + str( p_key - 1 ) ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MeanPoolingType() )
			else:
				raise Exception( "Invalid layer type specified" )

	else:
		raise Exception( "Invalid network configurations specified" )	
	return layer_configurations

if __name__ == "__main__":
	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
	input_layer_width = 6
	input_layer_height = 6
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 3
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs, 
		regular_weight_init_range, bias_weight_init_range, channels )

	for key in layer_configurations:
		print( "key: " + str( key ) )
		print( "value: " + str( layer_configurations[ key ] ) )
		print( "layer_configurations[ key ].get_super_type(): " + str( layer_configurations[ key ].get_super_type() ) )
		print( "layer_configurations[ key ].get_sub_type(): " + str( layer_configurations[ key ].get_sub_type() ) )
		print( "layer_configurations[ key ].get_channels(): " + str( layer_configurations[ key ].get_channels() ) )
		print( "layer_configurations[ key ].get_identity(): " + str( layer_configurations[ key ].get_identity() ) )
		print( "layer_configurations[ key ].get_regular_weight_init_range(): " + str( layer_configurations[ key ].get_regular_weight_init_range() ) )
		print( "layer_configurations[ key ].get_receptive_field_size(): " + str( layer_configurations[ key ].get_receptive_field_size() ) )
		print( "layer_configurations[ key ].get_width(): " + str( layer_configurations[ key ].get_width() ) )
		print( "layer_configurations[ key ].get_height(): " + str( layer_configurations[ key ].get_height() ) )
		print( "\n\n")
	
	