# -*- coding: UTF-8 -*-
import types
from convolutional_network_types import * 
from network_validation_tools import *
import numpy as np
from layer_constraints import LayerConstraints
from convolutional_network_types import *
from layers import *
from data_formatting import *
from v2 import EdgeGroup
from build_network import compute_convolutional_layer_depth
from build_network import compute_convolutional_layer_breadth


"""
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
"""

ROW_STEP = 1
ALONG_VERTICAL_AXIS = 0
APPEND_ROW = 0
VERTICAL_DIMENSION = 0
COLUMN_MEAN = 1

class ConvolutionalNetwork:
	def __init__( self, layer_constraints, 
		learning_rate, identity ):
		self.layer_constraints = layer_constraints
		self.learning_rate = learning_rate
		self.identity = identity
		self.layers = []

	def assemble_network( self ):
		for layer in self.layer_constraints:
			if ( isinstance( self.layer_constraints[layer].get_super_type(), 
				InputType ) ):
				self.layers.append( InputLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(), self.layer_constraints[layer].get_receptive_field_size(), [1.0]*self.layer_constraints[ layer ].get_channels() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				ConvolutionalType ) ):
				self.layers.append( ConvolutionalLayer(self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				OutputType ) ):
				self.layers.append( OutputLayer( self.layer_constraints[layer].get_identity(), 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(),
					self.layer_constraints[ layer ].get_channels(), 
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				FullyConnectedType ) ):
				self.layers.append( FullyConnectedLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

			elif ( isinstance( self.layer_constraints[layer].get_super_type(), 
				PoolingType ) ):
				if ( isinstance( self.layer_constraints[layer].get_sub_type(), 
					MinPoolingType ) ):
					self.layers.append( MaxPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(), 
					MeanPoolingType ) ):
					self.layers.append( MeanPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) ) 
				elif ( isinstance( self.layer_constraints[layer].get_sub_type(),
					MaxPoolingType ) ):
					self.layers.append( MaxPoolingLayer( self.layer_constraints[layer].get_identity() , 
					self.layer_constraints[layer].get_width(), self.layer_constraints[layer].get_height(), 
					self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_receptive_field_size(),
					[1.0]*self.layer_constraints[ layer ].get_channels(),
					self.layer_constraints[layer].get_regular_weight_init_range(), 
					self.layer_constraints[layer].get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

				else:
					raise Exception( "Invalid pooling type specified as sub-type of PoolingType" ) 
			else:
				raise Exception( "Invalid LayerConstraints object specified" )
			if len( self.layers ) != 1:
				self.layers[ len( self.layers ) - 1 ].assemble_layer()

	# Returns a dict of matrices, where 
	# the number of dict elements denote
	# the number of input channels. 
	# The key of the dict is equivalent to 
	# the corresponding channel; for example, 
	# key '0' denotes the channel 0. 
	# The keys are of type int. 
	def hypothesis( self, inputs ):
		for layer in range( len( self.layers ) ):
			if layer == 0:
				for channel in range( self.layers[layer].get_channels() ):
					self.layers[layer].set_regular_activations( inputs[ channel : np.shape( inputs )[VERTICAL_DIMENSION] : self.layers[layer].get_channels()], channel )
			elif layer == 1: 
				computed_activations = [np.empty(( self.layers[layer].get_height() * self.layers[layer].get_width(),
					np.shape( inputs )[1] ))] * self.layers[layer].get_channels()
				for channel in range( self.layers[layer].get_channels() ):
					indices = self.layers[layer-1].get_indices_model(channel)
					for row, col in self.iterate_over_input_groups( 0, 
						self.layers[layer].get_input_width(), 
						self.layers[layer].get_input_height(), self.layers[layer].get_rfs() ):
						if isinstance( self.layers[layer], ConvolutionalLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.sum( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MaxPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.amax( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MinPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
								np.amin( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MeanPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.mean( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], FullyConnectedLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
								self.layers[layer].get_regular_activations( channel ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )	

						computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, inputs ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

					self.layers[layer].set_regular_activations( computed_activations[channel] + 
						np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
						self.layers[layer].get_bias_weights( channel ) ) ) , channel )

			elif layer > 1:
				computed_activations = [np.empty(( self.layers[layer].get_height() * self.layers[layer].get_width(),
					np.shape( inputs )[1] ))] * self.layers[layer].get_channels()
				for channel in range( self.layers[layer].get_channels() ):
					indices = self.layers[layer].get_indices_model( None )
					
					for row, col in self.iterate_over_input_groups( 0, 
						self.layers[layer].get_input_width(), 
						self.layers[layer].get_input_height(), self.layers[layer].get_rfs() ):
						if isinstance( self.layers[layer], ConvolutionalLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.sum( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MaxPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.amax( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MinPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.amin( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel) ) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], MeanPoolingLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							np.mean( extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)) ), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )

						elif isinstance( self.layers[layer], FullyConnectedLayer ):
							computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							self.layers[layer-1].get_regular_activations(channel), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )		

						"""
						Must collate regular activations to form 
						a matrix composed of all input channels, 
						rather than channel-specific matrices. 
						Otherwise, write an alternative implementation. 
						"""
						computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )
					self.layers[layer].set_regular_activations( computed_activations[channel] + 
						np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
						self.layers[layer].get_bias_weights( channel ) ) ) , channel ) 
		return self.layers[layer].get_regular_activations()




	def back_propagate( self, inputs, target_outputs ):
		node_errors = {}
		provisional_node_errors = {}
		regular_weight_gradients = {}
		bias_weight_gradients = {}

		for layer in range( 1, len( self.layers ) ):
			node_errors[layer] = {}
			regular_weight_gradients[layer] = {}
			for channel in range( self.layers[layer].get_channels() ):
				node_errors[layer][channel] = np.asmatrix( np.zeros(( self.layers[layer].get_height() * self.layers[layer].get_width(), 1 ) ) )
				regular_weight_gradients[layer][channel] = np.asmatrix( np.zeros((self.layers[layer].get_height() * self.layers[layer].get_width(), 1 ) ) )
		
			bias_weight_gradients[layer] = {}

		for layer in range( len( self.layers ) - 1, 0, -1 ):
			for channel in range( self.layers[layer].get_channels() ):
				indices = np.asmatrix( np.arange( self.layers[layer].get_width() 
					* self.layers[layer].get_height() ).reshape(( self.layers[layer].get_height(), 
						self.layers[layer].get_width() )) )
				if ( layer == len( self.layers ) - 1 ):
					if isinstance( self.layers[layer], ConvolutionalLayer ):
						# something like along these lines..
						pass

					elif isinstance( self.layers[layer], MaxPoolingLayer ):
						pass
					elif isinstance( self.layers[layer], MinPoolingLayer ):
						pass
					elif isinstance( self.layers[layer], MeanPoolingLayer ):
						pass
					elif isinstance( self.layers[layer], FullyConnectedLayer ):
						pass
					elif isinstance( self.layers[layer], OutputLayer ):
						pass
					
					node_errors[ layer ][channel] = np.sum( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )
					for row in range( self.layers[layer].get_height() ):
						for col in range( self.layers[layer].get_width() ):
							regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
								np.sum( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height() , self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
								col : col + self.layers[layer].get_rfs() ] ), node_errors[layer][channel][row * self.layers[layer].get_width() + col ] )



				else:
					if isinstance( self.layers[layer], InputLayer ):
						pass
					elif isinstance( self.layers[layer], ConvolutionalLayer ):
				
						for row in range( self.layers[layer].get_height() ):
							for col in range( self.layers[layer].get_width() ):
								rfs = self.layers[layer+1].get_rfs()
								next_layer_height = self.layers[layer+1].get_height()
								next_layer_width = self.layers[layer+1].get_width()
								if ( ( (row - (rfs - 1)) >= 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = row - (rfs - 1)
									col_from = col - (rfs - 1) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									# compute_hidden_error( self, activations, errors, weights )
									
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.sum( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) < 0 ) ):
									row_from = 0 
									col_from = 0 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.sum( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = 0 
									col_from = col - ( rfs - 1 ) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.sum( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
								else:
									row_from = row - (rfs - 1) 
									col_from = 0
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width
									else:
										col_to = col + 1
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.sum( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									 
								regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
									np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
										self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
									col : col + self.layers[layer].get_rfs() ] ),
										node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )



					elif isinstance( self.layers[layer], MaxPoolingLayer ):
						for row in range( self.layers[layer].get_height() ):
							for col in range( self.layers[layer].get_width() ):
								rfs = self.layers[layer+1].get_rfs()
								next_layer_height = self.layers[layer+1].get_height()
								next_layer_width = self.layers[layer+1].get_width()
								if ( ( (row - (rfs - 1)) >= 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = row - (rfs - 1)
									col_from = col - (rfs - 1) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 

									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.max( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
									
								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) < 0 ) ):
									row_from = 0 
									col_from = 0 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.max( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = 0 
									col_from = col - ( rfs - 1 ) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.max( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
								else:
									row_from = row - (rfs - 1) 
									col_from = 0
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width
									else:
										col_to = col + 1
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.max( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
								regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
									np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
										self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
									col : col + self.layers[layer].get_rfs() ] ),
										node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )

						
	
					elif isinstance( self.layers[layer], MinPoolingLayer ):
						
						for row in range( self.layers[layer].get_height() ):
							for col in range( self.layers[layer].get_width() ):
								rfs = self.layers[layer+1].get_rfs()
								next_layer_height = self.layers[layer+1].get_height()
								next_layer_width = self.layers[layer+1].get_width()
								if ( ( (row - (rfs - 1)) >= 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = row - (rfs - 1)
									col_from = col - (rfs - 1) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 

									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.min( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) < 0 ) ):
									row_from = 0 
									col_from = 0 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.min( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = 0 
									col_from = col - ( rfs - 1 ) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.min( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
								else:
									row_from = row - (rfs - 1) 
									col_from = 0
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width
									else:
										col_to = col + 1
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.min( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
									
								regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
									np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
										self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
									col : col + self.layers[layer].get_rfs() ] ),
										node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )


					elif isinstance( self.layers[layer], MeanPoolingLayer ):
						
						for row in range( self.layers[layer].get_height() ):
							for col in range( self.layers[layer].get_width() ):
								rfs = self.layers[layer+1].get_rfs()
								next_layer_height = self.layers[layer+1].get_height()
								next_layer_width = self.layers[layer+1].get_width()
								if ( ( (row - (rfs - 1)) >= 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = row - (rfs - 1)
									col_from = col - (rfs - 1) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 

									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.mean( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
								
									
								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) < 0 ) ):
									row_from = 0 
									col_from = 0 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.mean( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
						

								elif ( ( (row - (rfs - 1)) < 0 ) & ( (col - (rfs - 1)) >= 0 ) ):
									row_from = 0 
									col_from = col - ( rfs - 1 ) 
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width 
									else:
										col_to = col + 1 
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.mean( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
							
								else:
									row_from = row - (rfs - 1) 
									col_from = 0
									if ( row >= next_layer_height ):
										row_to = next_layer_height 
									else:
										row_to = row + 1 
									if ( col >= next_layer_width ):
										col_to = next_layer_width
									else:
										col_to = col + 1
									node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_hidden_error( np.take( self.layers[layer].get_regular_activations( channel ), [ row * self.layers[layer].get_width() + col ] ), np.mean( 
										node_errors[ layer + 1 ][ channel ].reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ), np.mean( self.layers[layer+1].get_regular_weights( channel ).reshape( ( next_layer_height, next_layer_width ) )[ row_from : row_to,
											col_from : col_to ] ) )
						
								regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
									np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
										self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
									col : col + self.layers[layer].get_rfs() ] ),
										node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )		

			
					elif isinstance( self.layers[layer], FullyConnectedLayer ):
						pass
					elif isinstance( self.layers[layer], OutputLayer ):
						pass
		
			
				if ( (layer+1) <= ( len( self.layers ) - 1 ) ):
					bias_weight_gradients[layer][channel] = np.transpose( self.compute_bias_weight_gradients( node_errors[layer+1][channel] ) )
		
				
				
				
				self.layers[layer].set_regular_weight_changes( 
					self.layers[layer].get_regular_weight_changes( channel ) + np.transpose( regular_weight_gradients[layer][channel] ), channel )
				
	
	def iterate_over_input_groups( self, start, input_width, input_height, rfs ):
		for row in range( input_height - rfs + 1 ):
			for col in range( input_width - rfs + 1 ):
				yield row, col


	def get_layers( self ):
		return self.layers

	# param: activations
	# @param: errors
	# @param: weights
	def compute_hidden_error( self, activations, errors, weights ):
		return np.multiply( np.dot( weights, errors ), 
			np.multiply( activations, ( 1 - activations ) ) )

	# @param: hypothesis
	# @param: target
	# Dimensionalities of both parameters must be equal; i.e., 
	# assert( numpy.shape( hypothesis ) == numpy.shape( target ) )
	def compute_output_error( self, hypothesis, target ):
		assert( np.shape( hypothesis ) == np.shape( target ) )
		return np.multiply( - ( target - hypothesis ), np.multiply( hypothesis, ( 1 - hypothesis ) ) )

	def compute_regular_weight_gradients( self, activations, errors ):
		return np.multiply( errors, activations )
		

	def compute_bias_weight_gradients( self, errors ):
		return errors
		
	# this concerns pre-programming aspects
	# such as randomisation of batches
	def preprocess_inputs( self ):
		pass



if __name__ == "__main__":

	# 'input_layer_width'			 : int
	# 'input_layer_height' 			 : int
	# 'proposed_layer_types_and_rfs' : dict of dicts, where the key of the 
	# primary dict denotes the layer index, and the value is a sub-dict, where
	# the key of the sub-dict is a layer object type, and the corresponding value
	# denotes the proposed receptive field size, passed as an int-type value. 

	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 2 }, 2 : { MeanPoolingType() : 2 }, 
		3 : { ConvolutionalType() : 1 } }
	input_layer_width = 3
	input_layer_height = 3
	instances_per_batch = 3
	regular_weight_init_range = [0.1,0.2]
	bias_weight_init_range = [0.1,0.2]
	channels = 3

	# returns a one-dimensional key-value dict. 
	# key 'convention' is 'layer x', 
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
		regular_weight_init_range, bias_weight_init_range, channels )
	network = ConvolutionalNetwork( layer_configurations, 0.1, "Test Conv. S.N.N." )
	network.assemble_network()

	inputs = np.ones(( 
		input_layer_width * input_layer_height * channels ) * instances_per_batch )
	inputs = np.asmatrix( inputs.reshape( input_layer_width * input_layer_height * channels, instances_per_batch ) )
	targets = np.asmatrix( np.ones( (layer_configurations['layer 3'].get_height() * layer_configurations[ 'layer 3' ].get_width(), 
		instances_per_batch ) ) )
	inputs = inputs / 20.0

	print( "inputs: " + str( inputs ) )


	for i in range( input_layer_width * input_layer_height * channels ):
		inputs[i].fill( i + 1 )
	print( "hypothesis: " + str( network.hypothesis( inputs ) ) )
	network.back_propagate( inputs, targets )

	for layer in range(1, len( network.get_layers() ) ):
		print( "network.layers[layer].get_regular_weights(): " + 
			str( network.layers[layer].get_regular_weights(0) ) )

	"""
	for layer in range( 1, len( network.get_layers() ) ):
		print( "network.layers[layer].get_regular_weight_changes(0): " + str( 
			network.layers[layer].get_regular_weight_changes(0) ) )
	"""

	
	



