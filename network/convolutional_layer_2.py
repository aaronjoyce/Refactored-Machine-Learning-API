import numpy as np

from v2 import Layer
from v2 import Node
from v2 import BiasNode


class ConvolutionalLayer( Layer ):
	SHARED_WEIGHTS = 1
	TYPE = 0
	def __init__( self, identity, 
		input_dimensionality, 
		regular_weight_init_range, 
		bias_weight_init_range, 
		local_receptive_field_size, 
		channels, overlap ):
		Layer.__init__( self, identity, 
			input_dimensionality, regular_weight_init_range, 
			compute_convolutional_layer_depth( input_dimensionality[1], 
				local_receptive_field_size, overlap ), 
			compute_convolutional_layer_breadth( input_dimensionality[0], 
				local_receptive_field_size, overlap ),
			 bias_weight_init_range )
		self.channels = channels
		self.local_receptive_field_size = local_receptive_field_size
		self.nodes = np.empty( (channels, self.depth * self.breadth ), dtype=object)
		self.overlap = overlap

	def build_layer( self ):
		for channel in range( self.channels ):
			for node in range( self.depth * self.breadth ):
				self.nodes[channel][node] = Node( self.SHARED_WEIGHTS, node % self.breadth, node / self.breadth,
					self.SHARED_WEIGHTS, self.regular_weight_init_range )

		self.bias_node = BiasNode( None, node % self.breadth, node / self.breadth, 1.0 )


	# returns a column vector
	def get_regular_weights( self, channel ):
		temp = np.empty( ( 1, self.depth * self.breadth ) )
		for node in range( self.depth * self.breadth ):
			temp[0][node] = self.nodes[channel][node].get_regular_weights()
		return np.transpose( temp )

	def get_all_regular_weight_changes( self, channel ):
		temp = np.empty( ( 1, self.depth * self.breadth ) )
		for node in range( self.depth * self.breadth ):
			temp[0][node] = self.nodes[channel][node].get_regular_weight_changes()
		return np.transpose( temp ) 

	def set_all_regular_weight_changes( self, weight_changes, channel ):
		for node in range( self.depth * self.breadth ):
			self.nodes[channel][node].set_regular_weight_changes( np.asmatrix( weight_changes[ node ] ) )


	def set_all_regular_weights( self, weights, channel ):
		for node in range( self.depth * self.breadth ):
			self.nodes[ channel ][ node ].set_regular_weights( 
				weights[ node ] )
	def get_type( self ):
		return self.TYPE

	def update_all_node_activations( self, activations, channel ):
		activations = [activations]
		for node in range( self.depth * self.breadth ):
			self.nodes[channel][node].set_node( np.take( activations, [node], 1 ) )

	def get_all_node_activations( self, channel ):
		temp = []
		for node in range( self.depth * self.breadth ):
			temp.append( self.nodes[channel][node].get_node() )
		return np.transpose( temp )[0]

	def get_all_nodes( self ):
		return self.nodes

	def get_nodes( channel ):
		return self.nodes[channel]

	def total_channels( self ):
		return self.channels

# mark.little@storyful.com


		# instance attributes
# old method
"""
def compute_convolutional_layer_depth( input_depth, 
	receptive_field ):
	return int( np.ceil( float(input_depth)/receptive_field))

# old method 
def compute_convolutional_layer_breadth( input_breadth, 
	receptive_field ):
	return int( np.ceil( float(input_breadth)/receptive_field))

"""

def compute_convolutional_layer_depth( input_depth, 
	receptive_field, step ):
	if ( is_odd( input_depth ) & ( is_odd( receptive_field ) == False ) ):
		assert is_odd( step )
	assert receptive_field >= step
	if ( is_odd( input_depth) & ( is_odd( receptive_field ) ) ):
		assert step <= 2 
	if ( ( is_odd( input_depth ) == False ) & is_odd( receptive_field) ):
		assert is_odd( step )
	if ( is_odd( input_depth) & ( is_odd( receptive_field ) == False ) ):
		assert is_odd( step )
	return (input_depth - receptive_field)/step + 1

def compute_convolutional_layer_breadth( input_breadth, 
	receptive_field, step ):
	if ( is_odd( input_breadth ) & ( is_odd( receptive_field ) == False ) ):
		assert is_odd( step )
	assert receptive_field >= step
	if ( is_odd( input_breadth ) & ( is_odd( receptive_field ) ) ):
		assert step <= 2 
	if ( ( is_odd( input_breadth ) == False ) & is_odd( receptive_field) ):
		assert is_odd( step )
	if ( is_odd( input_breadth) & ( is_odd( receptive_field ) == False ) ):
		assert is_odd( step )
	return (input_breadth - receptive_field)/step + 1

def is_dimensionality_compatible( image_size, receptive_field_radii ):

	compatible= True
	running_dimensionality = float( image_size )
	index = 0

	while ( compatible & (index < len( receptive_field_radii ) )):
		if ( running_dimensionality / receptive_field_radii[index] >= 1 ):
			print "index: " + str( index )
			print "np.ceil( running_dimensionality / receptive_field_radii[index] ): " + str( 
				np.ceil( running_dimensionality / receptive_field_radii[index] ) )
			running_dimensionality = np.ceil( running_dimensionality / receptive_field_radii[index] )
			index += 1
		else:
			return False
	return True

def is_odd(num):
    return num & 0x1

if __name__ == "__main__":
	"""
	convolutional_layer = ConvolutionalLayer( "Layer 0", 
		[6,6], [0.1,0.2],[0.1,0.2], 3, 3, 0)
	convolutional_layer.build_layer()
	print "convolutional_layer: " + str( convolutional_layer.nodes )
	# this appears to be working, but it warrants further testing
	print "convolutional_layer.get_regular_weights( 0 ): " + str( convolutional_layer.get_regular_weights( 0 ) )
	print "convolutional_layer.get_regular_weights( 1 ): " + str( convolutional_layer.get_regular_weights( 1 ) )
	print "convolutional_layer.get_regular_weights( 2 ): " + str( convolutional_layer.get_regular_weights( 2 ) )
	"""
