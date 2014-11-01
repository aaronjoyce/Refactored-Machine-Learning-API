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
	for primary_key in proposed_layer_types_and_rfs:
		for sub_key in proposed_layer_types_and_rfs[ primary_key ]:
			if ( proposed_layer_types_and_rfs[ primary_key ][ sub_key ] != 0 ):
				rfr.append( proposed_layer_types_and_rfs[primary_key][sub_key] )

	if ( universal_is_dimensionality_compatible( input_layer_width, rfr, 1 ) ):
		layer_configurations = {}
		for p_key in range( len( proposed_layer_types_and_rfs ) ):
			print( "p_key " + str( p_key ) )
			print( "proposed_layer_types_and_rfs[ p_key ]: " + str( 
				proposed_layer_types_and_rfs[ p_key ].keys()[0] ) )
			if ( isinstance( proposed_layer_types_and_rfs[ p_key ].keys()[0], InputType ) ):
				print( "is instance of" )

if __name__ == "__main__":
	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 4 }, 2 : { MaxPooling() : 2 }, 
		3 : { FullyConnectedType() : 2 } }
	input_layer_width = 6
	input_layer_height = 6
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 3
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs, 
		regular_weight_init_range, bias_weight_init_range, channels )
	