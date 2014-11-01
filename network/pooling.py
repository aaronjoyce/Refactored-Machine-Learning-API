import numpy as np

from v2 import Layer
from convolutional_layer_2 import ConvolutionalLayer

class PoolingLayer( ConvolutionalLayer ):
	MEAN_POOLING = 0
	MAX_POOLING = 1
	MIN_POOLING = 2
	TYPE = 1
	def __init__( self, summarisation_type, input_dimensionality, 
		receptive_field, channels, identity, regular_weight_init_range, 
		bias_weight_init_range, overlap ):
		ConvolutionalLayer.__init__( self, identity, input_dimensionality, 
			regular_weight_init_range, 
			bias_weight_init_range, receptive_field, channels, overlap )

		#ConvolutionalLayer.__init__( self, identity, input_dimensionality, 
		#	regular_weight_init_range, compute_pooling_layer_depth( input_dimensionality[1], receptive_field ), 
		#	compute_pooling_layer_breadth( input_dimensionality[0], receptive_field ), 
		#	bias_weight_init_range, receptive_field, channels, overlap )

		self.type = summarisation_type
		self.receptive_field = receptive_field
		self.channels = channels

	def type( self ):
		return self.TYPE

	def get_summarisation_type( self ):
		return self.type

def compute_pooling_layer_depth( input_depth, receptive_field ):
	return int( np.ceil( float( input_depth )/ receptive_field ) )

def compute_pooling_layer_breadth( input_breadth, receptive_field ):
	return int( np.ceil( float( input_breadth) / receptive_field ) )


# dimensionality is a function of 'input_dimensionality' and
# 'receptive_field'


if __name__ == "__main__":
	pooling_layer = PoolingLayer( 0, [4,4], 
		2, 3, "Pooling Layer 1", [0.1,0.2], 
		[0.1,0.2], 0 )
	pooling_layer.build_layer()
	print pooling_layer.nodes
	print "depth: " + str( pooling_layer.get_depth() )
	print "breadth: " + str( pooling_layer.get_breadth() )
	print "channels: " + str( pooling_layer.get_regular_weights(0) )