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
	layer_configurations = []
	for primary_key in proposed_layer_types_and_rfs:
		for sub_key in proposed_layer_types_and_rfs[ primary_key ]:
			print( "primary_key: " + str( primary_key) )
			print( "sub_key: " + str( sub_key ) )
			print( "proposed_layer_types_and_rfs[p_key][s_key]: " + str( 
				proposed_layer_types_and_rfs[primary_key][sub_key] ) )
			if ( proposed_layer_types_and_rfs[ primary_key ][ sub_key ] != 0 ):
				rfr.append( proposed_layer_types_and_rfs[primary_key][sub_key] )
	print( "len( proposed_layer_types_and_rfs ): "  + str( len( proposed_layer_types_and_rfs ) ) )

	if ( universal_is_dimensionality_compatible( input_layer_width, rfr, 1 ) ):
		for p_key in range( len( proposed_layer_types_and_rfs ) ):
			if ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], InputType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
						input_layer_width, input_layer_height, s_key, channels, regular_weight_init_range, 
						bias_weight_init_range, InputType(), InputType() ) )
			
			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], OutputType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints(
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, OutputType(), FullyConnectedType() ) )	
	

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], FullyConnectedType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(),
						proposed_layer_types_and_rfs[ p_key ][ s_key], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(),
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, FullyConnectedType(), FullyConnectedType() ) )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], ConvolutionalType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, ConvolutionalType(), ConvolutionalType() ) )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MinPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(),
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(),
					proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MinPoolingType() ) )
				
			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MaxPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MaxPoolingType() ) )

			elif ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], MeanPoolingType ) ):
				for s_key in proposed_layer_types_and_rfs[ p_key ].keys():
					layer_configurations.append( LayerConstraints( 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 
					compute_convolutional_layer_breadth( layer_configurations[ len( layer_configurations ) - 1 ].get_width(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1 ), 
					compute_convolutional_layer_depth( layer_configurations[ len( layer_configurations ) - 1 ].get_height(), 
						proposed_layer_types_and_rfs[ p_key ][ s_key ], 1), s_key, channels, regular_weight_init_range, 
					bias_weight_init_range, PoolingType(), MeanPoolingType() ) )
			else:
				raise Exception( "Invalid layer type specified" )

	else:
		raise Exception( "Invalid network configurations specified" )	
	return layer_configurations

if __name__ == "__main__":
	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
	1 : { ConvolutionalType() : 5 }, 2 : { MaxPoolingType() : 2 }, 
	3 : { ConvolutionalType() : 5 }, 4 : { MaxPoolingType() : 2 }, 
	5 : { ConvolutionalType() : 5 }, 6 : { MaxPoolingType() : 2 }, 
	7 : { ConvolutionalType() : 5 }, 8 : { MaxPoolingType() : 2 } }

	input_layer_width = 28
	input_layer_height = 28
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 1
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs, 
		regular_weight_init_range, bias_weight_init_range, channels )
	layer_configurations.append( LayerConstraints( 1, 1, 10, "Classifier Layer", 
		1, [0.1,0.2], [0.1,0.2], FullyConnectedType(), FullyConnectedType() ) )

	for layer_config_index in range( len( layer_configurations ) ):
		print( "layer_config_index: " + str( layer_config_index ) )
		print( "layer_configurations[layer].super_type: " + str( 
			layer_configurations[ layer_config_index ].get_super_type() ) )
		print( "layer_configurations[layer].sub_type: " + str( 
			layer_configurations[ layer_config_index ].get_sub_type() ) )
		print( "\n\n" )

	
	