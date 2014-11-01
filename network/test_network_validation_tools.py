# network_validation_tools.py
import unittest
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


class TestGenerateLayerConfigurations( unittest.TestCase ):
	def setUp( self ):
		proposed_layer_types_and_rfs_1 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_1 = 6
		input_layer_height_1 = 6
		regular_weight_init_range_1 = [0.1,0.2]
		bias_weight_init_range_1 = [0.1,0.2]
		channels_1 = 3
		layer_configurations_1 = generate_layer_configurations( 
			input_layer_width_1, input_layer_height_1, proposed_layer_types_and_rfs_1, 
			regular_weight_init_range_1, bias_weight_init_range_1, channels_1 )

		proposed_layer_types_and_rfs_2 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_2 = 7
		input_layer_height_2 = 7
		regular_weight_init_range_2 = [0.1,0.2]
		bias_weight_init_range_2 = [0.1,0.2]
		channels_2 = 3
		layer_configurations_2 = generate_layer_configurations( 
			input_layer_width_2, input_layer_height_2, proposed_layer_types_and_rfs_2, 
			regular_weight_init_range_2, bias_weight_init_range_2, channels_2 )

		proposed_layer_types_and_rfs_3 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_3 = 2
		input_layer_height_3 = 2
		regular_weight_init_range_3 = [0.1,0.2]
		bias_weight_init_range_3 = [0.1,0.2]
		channels_3 = 3
		layer_configurations_3 = generate_layer_configurations( 
			input_layer_width_3, input_layer_height_3, proposed_layer_types_and_rfs_3, 
			regular_weight_init_range_3, bias_weight_init_range_3, channels_3 )

		proposed_layer_types_and_rfs_4 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_4 = 5
		input_layer_height_4 = 5
		regular_weight_init_range_4 = [0.1,0.2]
		bias_weight_init_range_4 = [0.1,0.2]
		channels_4 = 3
		layer_configurations_4 = generate_layer_configurations( 
			input_layer_width_4, input_layer_height_4, proposed_layer_types_and_rfs_4, 
			regular_weight_init_range_4, bias_weight_init_range_4, channels_4 )

		proposed_layer_types_and_rfs_5 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_5 = 4
		input_layer_height_5 = 4
		regular_weight_init_range_5 = [0.1,0.2]
		bias_weight_init_range_5 = [0.1,0.2]
		channels_5 = 3
		layer_configurations_5 = generate_layer_configurations( 
			input_layer_width_5, input_layer_height_5, proposed_layer_types_and_rfs_5, 
			regular_weight_init_range_5, bias_weight_init_range_5, channels_5 )

		proposed_layer_types_and_rfs_6 = { 0 : { InputType() : 0 }, 
			1 : { ConvolutionalType() : 3 }, 2 : { MaxPooling() : 3 }, 
			3 : { FullyConnectedType() : 2 } }
		input_layer_width_6 = 3
		input_layer_height_6 = 3
		regular_weight_init_range_6 = [0.1,0.2]
		bias_weight_init_range_6 = [0.1,0.2]
		channels_6 = 3
		layer_configurations_6 = generate_layer_configurations( 
			input_layer_width_6, input_layer_height_6, proposed_layer_types_and_rfs_6, 
			regular_weight_init_range_6, bias_weight_init_range_6, channels_6 )

	# write some TestCase. 
	# objective: 100% code coverage; > 90% problem coverage

	def test_get_super_type( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_get_idenity( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_get_sub_type( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_get_receptive_field_size( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_get_width( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_get_height( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_idenity( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_sub_type( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_super_type( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_receptive_field_size( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_width( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

	def test_set_height( self ):
		self.assertTrue( layer_configurations_1, )
		self.assertTrue( layer_configurations_2, )
		self.assertTrue( layer_configurations_3, )
		self.assertTrue( layer_configurations_4, )
		self.assertTrue( layer_configurations_5, )
		sefl.assertTrue( layer_configurations_6, )

if __name__ == "__main__":
	unittest.main()
