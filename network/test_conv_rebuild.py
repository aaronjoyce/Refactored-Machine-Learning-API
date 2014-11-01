import unittest
import conv_rebuild
import convolutional_network_types
import layers
import layer_constraints
import network_validation_tools
import types


class TestConvolutionalNetwork( unittest.TestCase ):
	def setUp( self ):
		# Network 1
		proposed_layer_types_and_rfs_1 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_1 = 6
		input_layer_height_1 = 6
		regular_weight_init_range_1 = [0.1,0.2]
		bias_weight_init_range_1 = [0.1,0.2]
		channels_1 = 3
		layer_configurations_1 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_1 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 1" )
		network_1.assemble_network()

		# Network 2
		proposed_layer_types_and_rfs_2 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_2 = 5
		input_layer_height_2 = 5
		regular_weight_init_range_2 = [0.1,0.2]
		bias_weight_init_range_2 = [0.1,0.2]
		channels_2 = 2
		layer_configurations_2 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_2 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 2" )
		network_2.assemble_network() 

		# Network 3
		proposed_layer_types_and_rfs_3 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_3 = 7
		input_layer_height_3 = 7
		regular_weight_init_range_3 = [0.1,0.2]
		bias_weight_init_range_3 = [0.1,0.2]
		channels_3 = 4
		layer_configurations_3 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_3 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 3" )
		network_3.assemble_network() 

		# Network 4
		proposed_layer_types_and_rfs_4 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_4 = 7
		input_layer_height_4 = 7
		regular_weight_init_range_4 = [0.1,0.2]
		bias_weight_init_range_4 = [0.1,0.2]
		channels_4 = 1
		layer_configurations_3 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_4 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 4" )
		network_4.assemble_network() 

		# Network 5
		proposed_layer_types_and_rfs_5 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_5 = 4
		input_layer_height_5 = 4
		regular_weight_init_range_5 = [0.1,0.2]
		bias_weight_init_range_5 = [0.1,0.2]
		channels_5 = 1
		layer_configurations_5 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_5 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 5" )
		network_5.assemble_network() 

		# Network 6
		proposed_layer_types_and_rfs_6 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		input_layer_width_6 = 2
		input_layer_height_6 = 2
		regular_weight_init_range_6 = [0.1,0.2]
		bias_weight_init_range_6 = [0.1,0.2]
		channels_6 = 3
		layer_configurations_6 = generate_layer_configurations( 
			input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
			regular_weight_init_range, bias_weight_init_range, channels )
		network_6 = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N. 5" )
		network_6.assemble_network() 


	def test_get_super_type( self ):
		for layer in range( len( network_1.get_layers() ) ):
			# insert some priori
			if ( True ):
				self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )


	def test_get_idenity( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_get_sub_type( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_get_receptive_field_size( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_get_width( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_get_height( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_idenity( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_sub_type( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_super_type( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_receptive_field_size( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_width( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )

	def test_set_height( self ):
		for layer in range( len( network_1.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_2.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_3.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_4.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_5.get_layers() ) ):
			self.assertEqual( )
		for layer in range( len( network_6.get_layers() ) ):
			self.assertEqual( )
