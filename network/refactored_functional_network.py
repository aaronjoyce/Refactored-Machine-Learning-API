# -*- coding: UTF-8 -*-
import types
from convolutional_network_types import * 
from network_validation_tools_1 import *
import numpy as np
from layer_constraints import LayerConstraints
from convolutional_network_types import *
from layers import *
from data_formatting import *
from v2 import EdgeGroup
from build_network import compute_convolutional_layer_depth
from build_network import compute_convolutional_layer_breadth
from mnist_loader import *
from v2 import *


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
			if ( isinstance( layer.get_super_type(), 
				InputType ) ):
				self.layers.append( InputLayer(layer.get_identity(), 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(), layer.get_receptive_field_size(), [1.0]*layer.get_channels() ) )
			elif ( isinstance( layer.get_super_type(), 
				ConvolutionalType ) ):
				self.layers.append( ConvolutionalLayer(layer.get_identity(), 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(),
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( layer.get_sub_type(), 
				OutputType ) ):
				self.layers.append( FullyConnectedLayer( layer.get_identity(), 
					layer.get_width(), layer.get_height(),
					layer.get_channels(), 
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
			elif ( isinstance( layer.get_super_type(), 
				FullyConnectedType ) ):
				self.layers.append( FullyConnectedLayer( layer.get_identity() , 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(),
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

			elif ( isinstance( layer.get_super_type(), 
				PoolingType ) ):
				if ( isinstance( layer.get_sub_type(), 
					MinPoolingType ) ):
					self.layers.append( MinPoolingLayer( layer.get_identity() , 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(),
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )
				elif ( isinstance( layer.get_sub_type(), 
					MeanPoolingType ) ):
					self.layers.append( MeanPoolingLayer( layer.get_identity() , 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(),
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) ) 
				elif ( isinstance( layer.get_sub_type(),
					MaxPoolingType ) ):
					self.layers.append( MaxPoolingLayer( layer.get_identity() , 
					layer.get_width(), layer.get_height(), 
					layer.get_channels(),
					layer.get_receptive_field_size(),
					[1.0]*layer.get_channels(),
					layer.get_regular_weight_init_range(), 
					layer.get_bias_weight_init_range(), 
					self.layers[ len( self.layers ) - 1 ].get_width(), 
					self.layers[ len( self.layers ) - 1 ].get_height() ) )

				else:
					raise Exception( "Invalid pooling type specified as sub-type of PoolingType" ) 
			else:
				raise Exception( "Invalid LayerConstraints object specified" )
			
			if len( self.layers ) != 1:
				self.layers[ len( self.layers ) - 1 ].assemble_layer()
		print( "Layer objects: " + str( self.layers ) )
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
						else:
							raise Exception( "Should not have been thrown" )
					"""
					self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel] + 
						np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
						self.layers[layer].get_bias_weights( channel ) ) ) ), channel )
					"""
					self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel ] + 
						np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
							self.layers[layer].get_bias_weights( channel ) ) ) ), channel )
					
				

			elif layer > 1:
				computed_activations = [np.empty(( self.layers[layer].get_height() * self.layers[layer].get_width(),
					np.shape( inputs )[1] ))] * self.layers[layer].get_channels()
				for channel in range( self.layers[layer].get_channels() ):
					indices = self.layers[layer].get_indices_model( None )
					
					if ( isinstance( self.layers[layer], FullyConnectedLayer ) ):
						computed_activations[channel] = np.transpose( np.dot(  
								np.transpose( self.layers[layer-1].get_regular_activations(channel) ),
								np.transpose( self.layers[layer].get_regular_weights(
								 channel ).reshape( ( self.layers[layer].get_height() * 
									self.layers[layer].get_width(), self.layers[layer].get_input_width() * self.layers[layer].get_input_height() ) ) ) ) )
					
						
						
					else:
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
								
							else:
								raise Exception( "Should not have been thrown" )

						"""
						Must collate regular activations to form 
						a matrix composed of all input channels, 
						rather than channel-specific matrices. 
						Otherwise, write an alternative implementation. 
						
						computed_activations[channel][ self.layers[layer].get_width() * row + col ] = np.sum( np.multiply( 
							extract_rows( ROW_STEP, self.layers[layer].get_rfs(), 
								row, col, indices, self.layers[layer-1].get_regular_activations(channel)), 
								np.take( self.layers[layer].get_regular_weights( channel ), 
									[ self.layers[layer].get_width() * row + col ] ) ), 0 )
						"""
					if ( layer == len( self.layers ) - 1 ):
						"""
						self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel] + 
							np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
							self.layers[layer].get_bias_weights( channel ) ) ) ), channel )
						"""
						self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel ] + 
							np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
							self.layers[layer].get_bias_weights( channel ) ) ) ), channel )
						
					else:
						
						self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel] + 
							np.transpose( np.multiply( self.layers[layer].get_bias_node( channel ), 
							self.layers[layer].get_bias_weights( channel ) ) ) ), channel )
						

						
						#self.layers[layer].set_regular_activations( sigmoid( computed_activations[channel] ), channel )
						

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
				if ( isinstance( self.layers[layer], FullyConnectedLayer ) == False ):
					regular_weight_gradients[layer][channel] = np.asmatrix( np.zeros((self.layers[layer].get_height() * self.layers[layer].get_width(), 1 ) ) )
				else:
					regular_weight_gradients[layer][channel] = np.asmatrix( np.zeros((self.layers[layer].get_height() * self.layers[layer].get_width() 
							* self.layers[layer-1].get_height() * self.layers[layer-1].get_width() ) ).reshape( ( self.layers[layer].get_width() * self.layers[layer].get_height(), 
							self.layers[layer-1].get_width() * self.layers[layer-1].get_height() ) ) )
			bias_weight_gradients[layer] = {}

		

		for layer in range( len( self.layers ) - 1, 0, -1 ):
			for channel in range( self.layers[layer].get_channels() ):
				indices = np.asmatrix( np.arange( self.layers[layer].get_width() 
					* self.layers[layer].get_height() ).reshape(( self.layers[layer].get_height(), 
						self.layers[layer].get_width() )) )
				if ( layer == len( self.layers ) - 1 ):
					if isinstance( self.layers[layer], ConvolutionalLayer ):
						# something like along these lines..
						node_errors[ layer ][channel] = np.sum( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )

					elif isinstance( self.layers[layer], MaxPoolingLayer ):
						node_errors[ layer ][channel] = np.max( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )
					elif isinstance( self.layers[layer], MinPoolingLayer ):
						node_errors[ layer ][channel] = np.min( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )
					elif isinstance( self.layers[layer], MeanPoolingLayer ):
						node_errors[ layer ][channel] = np.mean( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )
					elif isinstance( self.layers[layer], FullyConnectedLayer ):
						
						node_errors[ layer ][channel] = np.sum( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )
				
						
					elif isinstance( self.layers[layer], OutputLayer ):
						node_errors[ layer ][channel] = np.sum( self.compute_output_error( 
						self.layers[layer].get_regular_activations( channel ), target_outputs ), 1 )

					if ( isinstance( self.layers[layer], FullyConnectedLayer ) == True ):
						

						for row in range( node_errors[layer][channel].shape[0] ):
							regular_weight_gradients[layer][channel][row] = np.transpose( np.multiply( 
								np.sum( node_errors[layer][channel][row], 1 ), np.sum( self.layers[layer-1].get_regular_activations( channel ), 1) )  )
					
							
					else:
						
						for row in range( self.layers[layer].get_height() ):
							for col in range( self.layers[layer].get_width() ):
								regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
									np.sum( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height() , self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
									col : col + self.layers[layer].get_rfs() ] ), node_errors[layer][channel][row * self.layers[layer].get_width() + col ] )
					bias_weight_gradients[layer][channel] = np.transpose( node_errors[layer][channel] )
					
					if ( bias_weight_gradients[layer][channel].shape != self.layers[layer].get_bias_weight_changes( channel ).shape ):
						raise Exception( "Mismatch")
					self.layers[layer].set_bias_weight_changes( self.layers[layer].get_bias_weight_changes( channel ) + 
						bias_weight_gradients[layer][channel], channel )

				else:
					if isinstance( self.layers[layer], InputLayer ):
						pass
					elif isinstance( self.layers[layer], ConvolutionalLayer ):
					
						if ( isinstance( self.layers[layer+1], FullyConnectedLayer ) == False ):
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
									
							


						else:
							node_errors[layer][channel] = np.sum( self.compute_hidden_error( 
								self.layers[layer].get_regular_activations( channel ), node_errors[layer+1][channel], 
								self.layers[layer+1].get_regular_weights( channel ).reshape( ( self.layers[layer].get_height() * self.layers[layer].get_width(), 
									self.layers[layer+1].get_width() * self.layers[layer+1].get_height() ) ) ), 1 )
							for row in range( self.layers[layer].get_height() ):
								for col in range( self.layers[layer].get_width() ):
									regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
										np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
											self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
										col : col + self.layers[layer].get_rfs() ] ),
											node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )
						bias_weight_gradients[layer][channel] = np.transpose( node_errors[layer][channel] )
						
						if ( bias_weight_gradients[layer][channel].shape != self.layers[layer].get_bias_weight_changes( channel ).shape ):
							raise Exception( "Mismatch")
						self.layers[layer].set_bias_weight_changes( self.layers[layer].get_bias_weight_changes( channel ) + 
							bias_weight_gradients[layer][channel], channel )
			
							

					elif isinstance( self.layers[layer], MaxPoolingLayer ):
					
						if ( isinstance( self.layers[layer+1], FullyConnectedLayer ) == False ):
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
							
						else:
							node_errors[layer][channel] = np.sum( self.compute_hidden_error( 
								self.layers[layer].get_regular_activations( channel ), node_errors[layer+1][channel], 
								self.layers[layer+1].get_regular_weights( channel ).reshape( ( self.layers[layer].get_height() * self.layers[layer].get_width(), 
									self.layers[layer+1].get_width() * self.layers[layer+1].get_height() ) ) ), 1 )
							for row in range( self.layers[layer].get_height() ):
								for col in range( self.layers[layer].get_width() ):
									regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
										np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
											self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
										col : col + self.layers[layer].get_rfs() ] ),
											node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )
						bias_weight_gradients[layer][channel] = np.transpose( node_errors[layer][channel] )

						if ( bias_weight_gradients[layer][channel].shape != self.layers[layer].get_bias_weight_changes( channel ).shape ):
							raise Exception( "Mismatch")
						self.layers[layer].set_bias_weight_changes( self.layers[layer].get_bias_weight_changes( channel ) + 
							bias_weight_gradients[layer][channel], channel )



	
					elif isinstance( self.layers[layer], MinPoolingLayer ):
					
						if ( isinstance( self.layers[layer+1], FullyConnectedLayer ) == False ):
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
							
						
						else:
							node_errors[layer][channel] = np.sum( self.compute_hidden_error( 
								self.layers[layer].get_regular_activations( channel ), node_errors[layer+1][channel], 
								self.layers[layer+1].get_regular_weights( channel ).reshape( ( self.layers[layer].get_height() * self.layers[layer].get_width(), 
									self.layers[layer+1].get_width() * self.layers[layer+1].get_height() ) ) ), 1 )
							for row in range( self.layers[layer].get_height() ):
								for col in range( self.layers[layer].get_width() ):
									regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
										np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
											self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
										col : col + self.layers[layer].get_rfs() ] ),
											node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )
						bias_weight_gradients[layer][channel] = np.transpose( node_errors[layer][channel] )
					
						if ( bias_weight_gradients[layer][channel].shape != self.layers[layer].get_bias_weight_changes( channel ).shape ):
							raise Exception( "Mismatch")
						self.layers[layer].set_bias_weight_changes( self.layers[layer].get_bias_weight_changes( channel ) + 
							bias_weight_gradients[layer][channel], channel )



					elif isinstance( self.layers[layer], MeanPoolingLayer ):
						
						if ( isinstance( self.layers[layer+1], FullyConnectedLayer ) == False ):
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
							
						else:
							node_errors[layer][channel] = np.sum( self.compute_hidden_error( 
								self.layers[layer].get_regular_activations( channel ), node_errors[layer+1][channel], 
								self.layers[layer+1].get_regular_weights( channel ).reshape( ( self.layers[layer].get_height() * self.layers[layer].get_width(), 
									self.layers[layer+1].get_width() * self.layers[layer+1].get_height() ) ) ), 1 )
							for row in range( self.layers[layer].get_height() ):
								for col in range( self.layers[layer].get_width() ):
									regular_weight_gradients[layer][channel][ row * self.layers[layer].get_width() + col ] = self.compute_regular_weight_gradients( 
										np.mean( np.sum( self.layers[layer-1].get_regular_activations( channel ), 1 ).reshape( ( self.layers[layer-1].get_height(), 
											self.layers[layer-1].get_width() ) )[ row : row + self.layers[layer].get_rfs(), 
										col : col + self.layers[layer].get_rfs() ] ),
											node_errors[layer][channel][ row * self.layers[layer].get_width() + col ] )
						bias_weight_gradients[layer][channel] = np.transpose( node_errors[layer][channel] )
				
						if ( bias_weight_gradients[layer][channel].shape != self.layers[layer].get_bias_weight_changes( channel ).shape ):
							raise Exception( "Mismatch")
						self.layers[layer].set_bias_weight_changes( self.layers[layer].get_bias_weight_changes( channel ) + 
							bias_weight_gradients[layer][channel], channel )

						

										

			
					elif isinstance( self.layers[layer], FullyConnectedLayer ):
						print( "FullyConnectedLayer (back_propagate())" )
					elif isinstance( self.layers[layer], OutputLayer ):
						print( "OutputLayer (back_propagate())" )
						pass # must fill-in this 
					else:
						raise Exception( "This should not be raised" )

			
				if ( (layer+1) <= ( len( self.layers ) - 1 ) ):
					bias_weight_gradients[layer][channel] = np.transpose( self.compute_bias_weight_gradients( node_errors[layer+1][channel] ) )
				

				self.layers[layer].set_regular_weight_changes( 
					self.layers[layer].get_regular_weight_changes( channel ) + np.transpose( regular_weight_gradients[layer][channel] ).flatten(), channel )
				
				


		
	def train( self, inputs, target_outputs, epochs, batch_size, channels, weight_penalty ):
		
		#input_layer_dimensionality = [10,1]
		#hidden_layer_dimensionality = [[8,1]]
		#learning_rate = 3.0
		#bias_node = 1.0
		#regular_weight_init_range = [0.0,0.1]
		#bias_weight_init_range = [0.0,0.1]
		#weight_penalty = 0.01
		#output_dimensionality = [10,1]


		for epoch in range( epochs ):
			print( "epoch: " + str( epoch ) )
			for index in range( int( np.ceil( float( np.shape( inputs )[0] )/ batch_size ) ) ):
				input_subset = np.transpose( inputs[ index * batch_size : (index + 1 )*batch_size ] )
				output_subset = np.transpose( target_outputs[ index * batch_size : ( index + 1 ) * batch_size ] ) 
				self.hypothesis( input_subset )
				self.back_propagate( input_subset, output_subset )

			for l in range( 1, len( self.layers ) ):
				for c in range( channels ):
					self.layers[l].set_regular_weights( ( self.layers[l].get_regular_weights( c ) - 
						self.learning_rate * ( 1.0/int( np.ceil( float( np.shape( inputs )[0] )/ batch_size ) ) ) * 
						self.layers[l].get_regular_weight_changes( c ) + 
						weight_penalty * self.layers[l].get_regular_weights( c ) ), c )
					
					self.layers[l].set_bias_weights( ( self.layers[l].get_bias_weights( c ) - 
						self.learning_rate * ( 1.0/int( np.ceil( float( np.shape( inputs )[0] )/ batch_size ) ) ) * 
						self.layers[l].get_bias_weight_changes( c ) ), c )
					print( "denominator: " + str( int( np.ceil( float( np.shape( inputs )[0] )/ batch_size ) ) ) )
					print( "bias_weight_changes before: " + str( self.layers[l].get_bias_weight_changes( c ) ))
					self.layers[l].set_bias_weight_changes( np.asmatrix( np.zeros( ( 1, self.layers[l].get_bias_weight_changes( c ).size ) ) ), c )
					self.layers[l].set_regular_weight_changes( np.asmatrix( np.zeros( ( 1, self.layers[l].get_regular_weight_changes( c ).size ) ) ), c )
					print( "bias_weight_changes after: " + str( self.layers[l].get_bias_weight_changes( c ) ))




	
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
	"""
	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { MinPoolingType() : 7 }, 2 : { MinPoolingType() : 6 }, 
		3 : { MinPoolingType() : 6 }, 4 : { MinPoolingType() : 6 }, 
		5: { ConvolutionalType() : 4 } }
	"""
	
	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
	1 : { ConvolutionalType() : 2 }, 2 : { MeanPoolingType(): 2 },
	1 : { ConvolutionalType() : 2 }, 2 : { MeanPoolingType(): 2 },
	1 : { ConvolutionalType() : 2 }, 2 : { MeanPoolingType(): 2 } }
	
	input_layer_width = 28
	input_layer_height = 28
	instances_per_batch = 2
	out_dimensionality = 10
	data_instances = 200
	regular_weight_init_range = [0.0,0.05]
	bias_weight_init_range = [0.0,0.05]
	channels = 1

	# returns a one-dimensional key-value dict. 
	# key 'convention' is 'layer x', 
	layer_configurations = generate_layer_configurations( 
		input_layer_width, input_layer_height, proposed_layer_types_and_rfs,
		regular_weight_init_range, bias_weight_init_range, channels )
	layer_configurations.append( LayerConstraints( 1, 1, 
		out_dimensionality, "Classifier Layer", 
		1, [0.0,0.05], [0.0,0.05], FullyConnectedType(), FullyConnectedType() ) )


	network = ConvolutionalNetwork( layer_configurations, 3.0, "Test Conv. S.N.N." )
	network.assemble_network();
	
	batch_size = 1;
	epochs = 5;



"""
	inputs = np.empty(( 
		input_layer_width * input_layer_height * channels ) * instances_per_batch )
	inputs = np.asmatrix( inputs.reshape( input_layer_width * input_layer_height * channels, instances_per_batch ) )
	targets = np.asmatrix( np.ones( (layer_configurations[ len ( layer_configurations) - 1].get_height() * layer_configurations[ len( layer_configurations ) - 1 ].get_width(), 
		instances_per_batch ) ) )

	for i in range( input_layer_width * input_layer_height * channels ):
		inputs[i].fill( i + 1 )
	

	print( "regular weigths L1: " + str( network.layers[1].get_regular_weights( 0 ).shape ) )
	print( "regular weights L2: " + str( network.layers[2].get_regular_weights( 0 ).shape ) )
	print( "regular weights L3: " + str( network.layers[3].get_regular_weights( 0 ).shape ) )

	network.train( np.transpose( inputs ), np.transpose( targets ), epochs, batch_size, 1, 0.01 )



	
	for layer in range( len( network.layers ) ):
		if ( isinstance( network.layers[layer], InputLayer ) == False ):
			print( "layer: " + str( layer ) )
			print( "layers[layer]: " + str( network.layers[layer] ) )
			print( "layers[layer].get_width(): " + str( network.layers[layer].get_width() ) )
			print( "layers[layer].get_height(): " + str( network.layers[layer].get_height() ) )
			print( "layers[layer].get_input_height(): " + str( network.layers[layer].get_input_height() ) )
			print( "layers[layer].get_input_width(): " + str( network.layers[layer].get_input_width() ))
			print( "type: " + str( type( network.layers[layer].regular_weights ) ) )
			print( "len: " + str( len( network.layers[layer].regular_weights ) ) )
			print( "weights[ len( weights ) - 1 ]: " + str( network.layers[layer].regular_weights[ len( 
				network.layers[layer].regular_weights ) - 1 ].get_edges() ) )
	
"""	
	#number_of_instances = 20
    #output_dimensionality = 10


inputs = np.asmatrix( np.zeros( ( data_instances, input_layer_height * input_layer_width ) ) )
outputs = np.asmatrix( np.zeros( ( data_instances, out_dimensionality ) ) )

data = load_data_wrapper()
for instance in range( data_instances ):
        inputs[instance] = np.transpose( np.asmatrix( data[0][instance][0] ) )
        outputs[instance] = np.transpose( np.asmatrix( data[0][instance][1] ) )
    
      


network.train( inputs, outputs, epochs, batch_size, 1, 0.1 )


print( "target: " + str( np.transpose( outputs[0] ) ) )
print( "hypothesis: " + str( network.hypothesis( np.transpose( inputs[0]) ) ) )
print( "np.transpose( inputs[0]): " + str( np.transpose( inputs[0]) ) )
print( "\n")
print( "target: " + str( np.transpose( outputs[7] ) ) )
print( "hypothesis: " + str( network.hypothesis( np.transpose( inputs[7]) ) ) )
print( "np.transpose( inputs[7]): " + str( np.transpose( inputs[7]) ) )
print( "\n")
print( "target: " + str( np.transpose( outputs[25] ) ) )
print( "hypothesis: " + str( network.hypothesis( np.transpose( inputs[25]) ) ) )
print( "np.transpose( inputs[25]): " + str( np.transpose( inputs[25]) ) )
print( "\n")
print( "target: " + str( np.transpose( outputs[37] ) ) )
print( "hypothesis: " + str( network.hypothesis( np.transpose( inputs[37]) ) ) )
print( "np.transpose( inputs[37]): " + str( np.transpose( inputs[37]) ) )
print( "\n")


"""

for layer in range( len( network.layers ) ):
	for channel in range( network.layers[layer].get_channels() ):
		print( "layer: " + str( layer ) )
		print( "channel: " + str( channel ) )
		print( "activations: " + str( network.layers[layer].get_regular_activations( channel ) ) )

	proposed_layer_types_and_rfs = { 0 : { InputType() : 0 }, 
		1 : { ConvolutionalType() : 3 }, 2 : { MaxPoolingType() : 5 }, 
		3 : { ConvolutionalType() : 3 }, 4 : { FullyConnectedType() : 1 } }
	input_layer_width = 10
	input_layer_height = 10
	data_instances = 6
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


	inputs = np.empty(( 
		input_layer_width * input_layer_height * channels ) * instances_per_batch )
	inputs = np.asmatrix( inputs.reshape( input_layer_width * input_layer_height * channels, instances_per_batch ) )
	targets = np.asmatrix( np.ones( (layer_configurations['layer 3'].get_height() * layer_configurations[ 'layer 3' ].get_width(), 
		instances_per_batch ) ) )

	for i in range( input_layer_width * input_layer_height * channels ):
		inputs[i].fill( i + 1 )
	network.hypothesis( inputs )
	network.back_propagate( inputs, targets )
	

	inputs = np.empty( 
		( data_instances, input_layer_width * input_layer_height * channels ) )
	inputs.fill( 0.5 )
	inputs = np.asmatrix( inputs )
	targets = np.asmatrix( np.ones( ( data_instances, layer_configurations['layer 3'].get_height() * layer_configurations[ 'layer 3' ].get_width() ) ) )


	batch_size = 3
	epochs = 100
	test_data = np.empty( ( input_layer_width * input_layer_height * channels, 1 ))
	test_data.fill( 0.5 );
	test_data = np.asmatrix( test_data )
	network.train( inputs, targets, epochs, batch_size )
	print( "hypothesis: " + str( network.hypothesis( np.transpose( np.asmatrix( np.ones( ( input_layer_height * input_layer_width * channels ) ) ) ) ) ) )

"""
	
	
# Need to add a condition that check whether the next higher
# layer is a fully-connected layer, and if so, alter the node error
# and weight gradient functions, in order to utilise the large vector-like
# structures. 



