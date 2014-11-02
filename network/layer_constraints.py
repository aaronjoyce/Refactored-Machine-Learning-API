from convolutional_network_types import *
import types

# Super-type summary:
# 	InputType()
# 	ConvolutionalType()
# 	PoolingType()

# Sub-type summary:
#	NoSubtype() extends << InputType >>, << ConvolutionalType >>
#	MinPooling() extends << PoolingType >>
#	MeanPooling() extends << PoolingType >>
#	MaxPooling() extends << PoolingType >>



class LayerConstraints( object ):
	def __init__( self, receptive_field_size, 
		width, height, identity, channels, regular_weight_init_range, 
		bias_weight_init_range, super_type, sub_type = NoSubtype() ):
		self.receptive_field_size = receptive_field_size
		self.super_type = super_type
		assert isinstance( sub_type, type(super_type) )
		self.sub_type = sub_type
		self.width = width
		self.height = height
		self.identity = identity
		self.channels = channels
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range

	def get_super_type( self ):
		return self.super_type

	def get_channels( self ):
		return self.channels

	def get_identity( self ):
		return self.identity

	def get_regular_weight_init_range( self ):
		return self.regular_weight_init_range

	def get_bias_weight_init_range( self ):
		return self.bias_weight_init_range

	def get_sub_type( self ):
		return self.sub_type

	def get_receptive_field_size( self ):
		return self.receptive_field_size

	def get_width( self ):
		return self.width

	def get_height( self ):
		return self.height

	def set_super_type( self, super_type ):
		assert( isinstance( super_type, ConvolutionalType ) | 
			isinstance( super_type, PoolingType ) | 
			isinstance( super_type, InputType ) )
		self.super_type = super_type

	def set_idenity( self, identity ):
		self.identity = identity

	def set_sub_type( self, sub_type ):
		assert( isinstance( sub_type, type( self.super_type ) ) )
		self.subtype = sub_type

	def set_receptive_field_size( self, size ):
		self.receptive_field_size = size

	def set_width( self, width ):
		self.width = width

	def set_height( self, height ):
		self.height = height

	def set_channels( self, channels ):
		self.channels = channels

	def set_regular_weight_init_range( self, init_range ):
		self.regular_weight_init_range = init_range

	def set_bias_weight_init_range( self, init_range ):
		self.bias_weight_init_range = init_range



if __name__ == "__main__":
	layer_constraints = LayerConstraints( 3, 4, 4, "Layer 0", 3, [0.1,0.2], [0.1,0.2], ConvolutionalType() )
	print( "Super-type: " + str( layer_constraints.get_super_type() ) ) 
	print( "Sub-type: " + str( layer_constraints.get_sub_type() ) )
	print( "Receptive field size: " + str( layer_constraints.get_receptive_field_size() ) )
	print( "Width: " + str( layer_constraints.get_width() ) )
	print( "Height: " + str( layer_constraints.get_height() ) )

	layer_constraints.set_super_type( InputType() )
	layer_constraints.set_sub_type( NoSubtype() )
	print( "Super-type: " + str( layer_constraints.get_super_type() ) ) 
	print( "Sub-type: " + str( layer_constraints.get_sub_type() ) )




