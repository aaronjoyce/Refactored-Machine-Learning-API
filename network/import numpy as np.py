import types
from convolutional_network_types import * 

import numpy as np
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
import layer_constraints

class ConvolutionalLayer( Network ):
	def __init__( self, layer_constraints, 
		learning_rate, identity ):
		self.layer_constraints = layer_constraints
		self.learning_rate = learning_rate
		self.identity = identity

	def assemble_network( self ):
		for layer in self.layer_constraints:
			if ( isinstance( self.layer_constraints[layer].get_super_type(), 
				InputLayer ) ):
				pass
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				ConvolutionalLayer ) )
				pass
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				PoolingType ) )
				pass
			else:
				raise Exception( "Invalid LayerConstraints object specified" )

