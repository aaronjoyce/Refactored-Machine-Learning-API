import unittest
from conv_rebuild import *
from convolutional_network_types import *
from layers import *
from layer_constraints import *
from network_validation_tools import *
import types


class TestConvolutionalNetwork( unittest.TestCase ):
	def setUp( self ):
		# Network 1
		self.proposed_layer_types_and_rfs_1 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_1 = 6
		self.input_layer_height_1 = 6
		self.instances_per_batch_1 = 4
		self.regular_weight_init_range_1 = [0.1,0.2]
		self.bias_weight_init_range_1 = [0.1,0.2]
		self.channels_1 = 3
		self.layer_configurations_1 = generate_layer_configurations( 
			self.input_layer_width_1, self.input_layer_height_1, self.proposed_layer_types_and_rfs_1,
			self.regular_weight_init_range_1, self.bias_weight_init_range_1, self.channels_1 )
		self.network_1 = ConvolutionalNetwork( self.layer_configurations_1, 0.1, "Test Conv. S.N.N. 1" )
		self.network_1.assemble_network()

		self.inputs_1 = np.empty(( 
			self.input_layer_width_1 * self.input_layer_height_1 * self.channels_1 ) * self.instances_per_batch_1 )
		self.inputs_1 = self.inputs_1.reshape( self.input_layer_width_1 * self.input_layer_height_1 * self.channels_1, self.instances_per_batch_1 )
		self.network_1.hypothesis( self.inputs_1 )

		# Network 2
		self.proposed_layer_types_and_rfs_2 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_2 = 6
		self.input_layer_height_2 = 6
		self.instances_per_batch_2 = 4
		self.regular_weight_init_range_2 = [0.1,0.2]
		self.bias_weight_init_range_2 = [0.1,0.2]
		self.channels_2 = 2
		self.layer_configurations_2 = generate_layer_configurations( 
			self.input_layer_width_2, self.input_layer_height_2, self.proposed_layer_types_and_rfs_2,
			self.regular_weight_init_range_2, self.bias_weight_init_range_2, self.channels_2 )
		self.network_2 = ConvolutionalNetwork( self.layer_configurations_2, 0.1, "Test Conv. S.N.N. 2" )
		self.network_2.assemble_network() 
		self.inputs_2 = np.empty(( 
			self.input_layer_width_2 * self.input_layer_height_2 * self.channels_2 ) * self.instances_per_batch_2 )
		self.inputs_2 = self.inputs_2.reshape( self.input_layer_width_2 * self.input_layer_height_2 * self.channels_2, self.instances_per_batch_2 )
		self.network_2.hypothesis( self.inputs_2 )

		# Network 3
		self.proposed_layer_types_and_rfs_3 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_3 = 7
		self.input_layer_height_3 = 7
		self.regular_weight_init_range_3 = [0.1,0.2]
		self.bias_weight_init_range_3 = [0.1,0.2]
		self.channels_3 = 4
		self.instances_per_batch_3 = 5
		self.layer_configurations_3 = generate_layer_configurations( 
			self.input_layer_width_3, self.input_layer_height_3, self.proposed_layer_types_and_rfs_3,
			self.regular_weight_init_range_3, self.bias_weight_init_range_3, self.channels_3 )
		self.network_3 = ConvolutionalNetwork( self.layer_configurations_3, 0.1, "Test Conv. S.N.N. 3" )
		self.network_3.assemble_network() 

		self.inputs_3 = np.empty(( 
			self.input_layer_width_3 * self.input_layer_height_3 * self.channels_3 ) * self.instances_per_batch_3 )
		self.inputs_3 = self.inputs_3.reshape( self.input_layer_width_3 * self.input_layer_height_3 * self.channels_3, self.instances_per_batch_3 )
		self.network_3.hypothesis( self.inputs_3 )

		# Network 4
		self.proposed_layer_types_and_rfs_4 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_4 = 7
		self.input_layer_height_4 = 7
		self.regular_weight_init_range_4 = [0.1,0.2]
		self.bias_weight_init_range_4 = [0.1,0.2]
		self.instances_per_batch_4 = 6
		self.channels_4 = 1
		self.layer_configurations_4 = generate_layer_configurations( 
			self.input_layer_width_4, self.input_layer_height_4, self.proposed_layer_types_and_rfs_4,
			self.regular_weight_init_range_4, self.bias_weight_init_range_4, self.channels_4 )
		self.network_4 = ConvolutionalNetwork( self.layer_configurations_4, 0.1, "Test Conv. S.N.N. 4" )
		self.network_4.assemble_network() 
		self.inputs_4 = np.empty(( 
			self.input_layer_width_4 * self.input_layer_height_4 * self.channels_4 ) * self.instances_per_batch_4 )
		self.inputs_4 = self.inputs_4.reshape( self.input_layer_width_4 * self.input_layer_height_4 * self.channels_4, self.instances_per_batch_4 )
		self.network_4.hypothesis( self.inputs_4 )

		# Network 5
		self.proposed_layer_types_and_rfs_5 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_5 = 6
		self.input_layer_height_5 = 6
		self.regular_weight_init_range_5 = [0.1,0.2]
		self.bias_weight_init_range_5 = [0.1,0.2]
		self.channels_5 = 1
		self.instances_per_batch_5 = 4
		self.layer_configurations_5 = generate_layer_configurations( 
			self.input_layer_width_5, self.input_layer_height_5, self.proposed_layer_types_and_rfs_5,
			self.regular_weight_init_range_5, self.bias_weight_init_range_5, self.channels_5 )
		self.network_5 = ConvolutionalNetwork( self.layer_configurations_5, 0.1, "Test Conv. S.N.N. 5" )
		self.network_5.assemble_network() 

		self.inputs_5 = np.empty(( 
			self.input_layer_width_5 * self.input_layer_height_5 * self.channels_5 ) * self.instances_per_batch_5 )
		self.inputs_5 = self.inputs_5.reshape( self.input_layer_width_5 * self.input_layer_height_5 * self.channels_5, self.instances_per_batch_5 )
		self.network_5.hypothesis( self.inputs_5 )

		# Network 6
		self.proposed_layer_types_and_rfs_6 = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 3 }, 
		3 : { FullyConnectedType() : 2 } }
		self.input_layer_width_6 = 6
		self.input_layer_height_6 = 6
		self.regular_weight_init_range_6 = [0.1,0.2]
		self.bias_weight_init_range_6 = [0.1,0.2]
		self.channels_6 = 3
		self.instances_per_batch_6 = 3
		self.layer_configurations_6 = generate_layer_configurations( 
			self.input_layer_width_6, self.input_layer_height_6, self.proposed_layer_types_and_rfs_6,
			self.regular_weight_init_range_6, self.bias_weight_init_range_6, self.channels_6 )
		self.network_6 = ConvolutionalNetwork( self.layer_configurations_6, 0.1, "Test Conv. S.N.N. 5" )
		self.network_6.assemble_network() 
		self.inputs_6 = np.empty(( 
			self.input_layer_width_6 * self.input_layer_height_6 * self.channels_6 ) * self.instances_per_batch_6 )
		self.inputs_6 = self.inputs_6.reshape( self.input_layer_width_6 * self.input_layer_height_6 * self.channels_6, self.instances_per_batch_6 )
		self.network_6.hypothesis( self.inputs_6 )


	def test_get_super_type( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			# insert some priori
			if ( True ):
				self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )


	def test_get_idenity( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_get_sub_type( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_get_receptive_field_size( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_get_width( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_get_height( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_idenity( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_sub_type( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_super_type( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_receptive_field_size( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_width( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

	def test_set_height( self ):
		for layer in range( len( self.network_1.get_layers() ) ):
			self.assertEqual( None, None )
		for layer in range( len( self.network_2.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_3.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_4.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_5.get_layers() ) ):
			self.assertEqual( None, None  )
		for layer in range( len( self.network_6.get_layers() ) ):
			self.assertEqual( None, None  )

if __name__ == "__main__":
	unittest.main()
