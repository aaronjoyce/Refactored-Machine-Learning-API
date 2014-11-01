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


Y_AXIS_MEAN = 0
ORDER_0 = 0
ORDER_1 = 1
ORDER_2 = 2 
ORDER_3 = 3 
ORDER_4 = 4

def collate_reformed_inputs( inputs, size, depth, breadth ):
	temp = np.zeros( ((depth-size+1) * (breadth-size+1) ,size**2) )
	for row in range( depth - size + 1 ):
		for col in range( breadth - size + 1 ):
			temp[(row*compute_convolutional_layer_breadth( breadth, 
					size, 1 )) + col] = slicer( inputs, breadth, depth, col, row, size ).flatten()
	return temp;

def slicer( inputs, input_width, input_height, i, j, size ):
	assert i <= ( input_width - size )
	assert j <= ( input_height - size )
	assert input_width >= size
	assert input_height >= size
	#print( "row in slicer: " + str( i ) )
	#print( "col in slicer: " + str( j ) )
	#print( "size in slicer: " + str( size ) )
	temp = np.zeros((size,size))
	#print( "temp in slicer: " + str( temp ) )
	for row in range( size ):
		#print( "inputs section: " + str( 
		
		temp[row] = inputs[ ( row + j ) * input_width + i : (row+j)*input_width + size + i ]
	return temp

class ConvolutionalNetwork( Network ):
	MAPPING_LAYER = 0
	POOLING_LAYER = 1
	MEAN_POOLING = 0
	MAX_POOLING = 1
	MIN_POOLING = 2

	def __init__( self, input_channels, 
		input_layer_dimensionality, 
		pooling_types, learning_rate, regular_weight_init_range, 
		bias_weight_init_range, identification, order, local_receptive_fields, 
		weight_penalty ):
		Network.__init__( self, input_layer_dimensionality, None, learning_rate, 
			1.0, regular_weight_init_range, bias_weight_init_range, weight_penalty )
		self.input_channels = input_channels
		self.input_layer_dimensionality = input_layer_dimensionality
		self.pooling_types = pooling_types
		self.learning_rate = learning_rate
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range
		self.identification = identification
		self.order = order
		self.local_receptive_fields = local_receptive_fields
		# Used to store maximums, minimums, and/or means
		self.ancillary_storage = []
		self.layers = [] 


	def build_network( self ):
		for layer in range( len( self.order ) ):
			# if layer == 0, it must be a 
			# convolutional layer - pooling layers'
			# inputs can only be convolved features
			if ( layer == 0 ):
				self.layers.append( ConvolutionalLayer( 
					"Mapping Layer" + str( layer ), self.input_layer_dimensionality, 
					self.regular_weight_init_range, self.bias_weight_init_range, 
					self.local_receptive_fields[layer], self.input_channels, 1 ) )
			else:
				if ( self.order[layer][0] == self.MAPPING_LAYER ):
					self.layers.append( ConvolutionalLayer( "Mapping Layer " + str( layer ), 
						[self.layers[layer-1].get_breadth(), self.layers[layer-1].get_depth()], 
						self.regular_weight_init_range, self.bias_weight_init_range, 
						self.local_receptive_fields[layer], self.input_channels, 1 ) )
				elif ( self.order[layer][0] == self.POOLING_LAYER ):
					self.layers.append( PoolingLayer( 1, [self.layers[layer-1].get_breadth(), 
						self.layers[layer-1].get_breadth()], 
						self.local_receptive_fields[layer], self.input_channels, 
						("Pooling Layer " + str( layer )), self.regular_weight_init_range, 
						self.bias_weight_init_range, 1 ) )
					"""
					self.ancillary_storage.append( np.zeros( ( compute_pooling_layer_depth(
						self.layers[layer-1].get_depth(), self.local_receptive_fields[layer] ), 
						compute_pooling_layer_breadth( self.layers[layer-1].get_breadth(), 
							self.local_receptive_fields[layer] ) ) ) )	
					"""			
				else:
					raise UnrecognisedOrder( "Order specificed " + 
							"is neither a mapping layer nor a pooling layer")
			self.layers[layer].build_layer()


	def hypothesis( self, inputs ):

		reformed_inputs = []
		lrf_size = self.local_receptive_fields[0];
		self.ancillary_storage = []

		for layer in range( len( self.layers ) ):
			if layer == 0:
				for channel in range( self.input_channels ):
					print( "layer: " + str( layer ) )
					print( "channel: " + str( channel ) )
					reformed_inputs.append( inputs[channel : len( inputs ) : self.input_channels ] )
					self.ancillary_storage.append( np.sum( collate_reformed_inputs( reformed_inputs[channel], 
						lrf_size, self.input_layer_dimensionality[1], self.input_layer_dimensionality[0] ), 1 ) )
					
					self.layers[layer].update_all_node_activations( 
						np.asarray( np.multiply( np.transpose( np.asmatrix( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ] ) ), 
							self.layers[layer].get_regular_weights( channel ) ).flatten() )[0], channel )
					
					print( "weights: " + str( self.layers[layer].get_regular_weights( channel ) ) )
					print( "inputs: " + str( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ] ) )
					print( "result of product: " + str( self.layers[layer].get_all_node_activations( channel ) ) )
					print( "product result calculateed directly: " + str( 
						np.multiply( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ], 
							self.layers[layer].get_regular_weights( channel ) ).flatten()))
					print( "another product: " + str( 
						np.asarray( np.multiply( np.transpose( np.asmatrix( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ] ) ), 
							self.layers[layer].get_regular_weights( channel ) ).flatten() )[0] ) )
					print( "first arg: " + str( np.transpose( np.asmatrix( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ] ) ) ) )
					print( "second arg: " + str(self.layers[layer].get_regular_weights( channel ) ) )
					print( "\n\n\n\n\n")
			
			else: 
				for channel in range( self.input_channels ):
					if self.layers[layer].get_type() == self.MAPPING_LAYER:
						self.ancillary_storage.append( np.sum( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
								self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
								self.layers[layer-1].get_breadth() ), 1 ) )
						print( "layer: " + str( layer ) )
						print( "channel: " + str( channel ) )
						self.layers[layer].update_all_node_activations( 
							np.multiply( 
								self.ancillary_storage[ len( self.ancillary_storage ) - 1 ], 
								self.layers[layer].get_regular_weights( channel ) ).flatten(), channel )
						print( "direct result: " + str( np.multiply( 
								self.ancillary_storage[ len( self.ancillary_storage ) - 1 ], 
								self.layers[layer].get_regular_weights( channel ) ).flatten() ))
						print( "weights: " + str( self.layers[layer].get_regular_weights( channel ) ) )
						print( "inputs: " + str( self.ancillary_storage[ len( self.ancillary_storage ) - 1 ] ) )
						print( "result of product: " + str( self.layers[layer].get_all_node_activations( channel ) ) )
					elif self.layers[layer].get_type() == self.POOLING_LAYER:
						if self.layers[layer].get_summarisation_type() == self.MIN_POOLING:
							self.ancillary_storage.append( np.asarray( np.transpose( np.multiply( 
									np.amin( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1), 
									self.layers[layer].get_regular_weights( channel ) ) ) )[0] )
							print( "layer: " + str( layer ) )
							print( "channel: " + str( channel ) )
							print( "weights: " + str( self.layers[layer].get_regular_weights( channel ) ) )
							print( "inputs before summarisation: " + str( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ) ) )
							print( "inputs: " + str( np.amin( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1) ) )
							self.layers[layer].update_all_node_activations( 
								self.ancillary_storage[ len( self.ancillary_storage ) - 1], channel );
							print( "result of product: " + str( self.layers[layer].get_all_node_activations( channel ) ) ) 
							print( "\n\n\n\n\n")
							# np.amin( )
						elif self.layers[layer].get_summarisation_type() == self.MAX_POOLING:
							self.ancillary_storage.append( np.asarray( np.transpose( np.multiply( 
									np.transpose( np.asmatrix( np.amax( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1 ) ) ), 
									self.layers[layer].get_regular_weights( channel ) ) ) )[0] )
							print( "layer: " + str( layer ) )
							print( "channel: " + str( channel ) )
							print( "weights: " + str( self.layers[layer].get_regular_weights( channel ) ) )
							print( "inputs before summarisation: " + str( np.asmatrix( np.amax( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1 ) ) ) )
							print( "inputs: " + str( np.amax( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1 ) ) )
							print( "self.ancillary_storage: " + str( np.asarray( np.transpose( self.ancillary_storage[ len( self.ancillary_storage ) - 1] ) )[0] ) )				
							self.layers[layer].update_all_node_activations( 
								self.ancillary_storage[ len( self.ancillary_storage ) - 1 ], channel );
							print( "result of product: " + str( self.layers[layer].get_all_node_activations( channel ) ) ) 
							print( "\n\n\n\n\n")
							# np.amax( )
						elif self.layers[layer].get_summarisation_type() == self.MEAN_POOLING:
							self.ancillary_storage.append( np.asarray( np.transpose( np.multiply( 
									np.mean( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1 ), 
									self.layers[layer].get_regular_weights( channel ) ) ) )[0] )
							print( "layer: " + str( layer ) )
							print( "channel: " + str( channel ) )
							print( "weights: " + str( self.layers[layer].get_regular_weights( channel ) ) )
							print( "inputs before summarisation: " + str( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ) ) )
							print( "inputs: " + str( np.mean( collate_reformed_inputs( self.layers[layer-1].get_all_node_activations( channel )[0], 
									self.local_receptive_fields[layer], self.layers[layer-1].get_depth(), 
									self.layers[layer-1].get_breadth() ), 1 ) ) )
							print( "self.ancillary_storage: " + str( self.ancillary_storage[ len( self.ancillary_storage ) - 1] ) )
							self.layers[layer].update_all_node_activations( 
								self.ancillary_storage[ len( self.ancillary_storage ) - 1 ], channel );
							print( "result of product: " + str( self.layers[layer].get_all_node_activations( channel ) ) ) 
							print( "\n\n\n\n\n")
							# np.mean( )
						else:
							raise Exception( "Unrecognised pooling get_type() specificed" )
					else:
						raise Exception( "Error" )
			#print( "layer: " + str( layer ) )
		print( "ancillary_storage: " + str( self.ancillary_storage) )
			#print( "\n\n\n\n" )
			
	# Must make some assumptions about inputs' form:
	# 1. It's 2-dimensional...?
	# 2. How are channels delineated?


	def back_propagate( self, inputs, target_output ):
		errors = []
		weight_gradients = []
		bias_weight_gradients = []
		for layer in range( len( self.layers ) ):
			errors.append( [0] * self.input_channels )
			weight_gradients.append( [0] * self.input_channels )
			bias_weight_gradients.append( [0] * self.input_channels )

	
		reformed_inputs = []
		for channel in range( self.input_channels ):
			reformed_inputs.append( inputs[channel : len( inputs ) : self.input_channels ] )
		

		lastOneAccessed = len( self.ancillary_storage )
		layer = len( self.layers ) - 1
		for channel in range( self.input_channels ):
			print( "layer: " + str( layer ) )
			print( "channel: " + str( channel ) )
			print( "input activations to final-layer errors: " + str( 
				self.layers[ layer ].get_all_node_activations( channel ) ) )
			print( "target output to final-layer errors: " + str( target_output ) )
			errors[ layer ][ channel ] = self.compute_output_error( 
				self.layers[ layer ].get_all_node_activations( channel ), target_output )
			print( "input to weight gradients function (1): " + str( 
				self.ancillary_storage[ lastOneAccessed - ( self.input_channels - channel ) ]) )
			print( "input to weight_gradients function (2): " + str( 
				np.transpose( np.asmatrix( errors[ layer ][ channel ] ) ) ) )
			print( "\n\n")
			weight_gradients[ layer ][ channel ] = self.compute_regular_weight_gradients( 
				self.ancillary_storage[ lastOneAccessed - ( self.input_channels - channel ) ], 
				np.transpose( np.asmatrix( errors[ layer ][ channel ] ) ) )
			print( "weight_gradients output: " + str( weight_gradients[ layer ][channel] ) )
			self.layers[layer].set_all_regular_weight_changes( 
				( self.layers[layer].get_all_regular_weight_changes( channel ) + 
					weight_gradients[ layer ][ channel ] ), channel )
		print( "\n\n\n\n\n\n" )
		lastOneAccessed -= 3
		layer -= 1

		while ( layer >= 0 ):
			for channel in range( self.input_channels ):
				print( "layer: " + str( layer ) )
				print( "channel: " + str( channel ) )
				errors[ layer ][ channel ] = np.mean( self.compute_hidden_error( 
					np.transpose( np.asmatrix( self.ancillary_storage[ (lastOneAccessed)- ( self.input_channels - channel ) ] ) ), 
					np.transpose( errors[ layer + 1 ][ channel ] ), 
					self.layers[ layer + 1  ].get_regular_weights( channel ) ), 1)
				print( "activations input to hidden error: " + str(  
					np.transpose( np.asmatrix( self.ancillary_storage[ (lastOneAccessed)- ( self.input_channels - channel ) ] ) ) ) )
				print( "errors input to hidden error: " + str( np.transpose( errors[ layer + 1 ][ channel ] ) ) )
				print( "weights input to hidden error: " + str( 
					self.layers[ layer + 1  ].get_regular_weights( channel ) ) )
				print( "errors output: " + str( errors[layer][channel] ) )
				
				print( "activations input to weight_gradients: " + str( 
					np.transpose( np.asmatrix( self.ancillary_storage[ (lastOneAccessed) 
						- ( self.input_channels - channel ) ] ) ) ) )
				print( "errors input to weight_gradients: " + str( 
					np.transpose( np.asmatrix( errors[ layer ][ channel ] ) )) )
				weight_gradients[ layer ][ channel ] = self.compute_regular_weight_gradients( 
					np.transpose( np.asmatrix( self.ancillary_storage[ (lastOneAccessed) 
						- ( self.input_channels - channel ) ] ) ), 
					np.transpose( np.asmatrix( errors[ layer ][ channel ] ) ) )	
				print( "weight_gradients output: " + str( 
					weight_gradients[layer][channel] ) )
				self.layers[ layer ].set_all_regular_weight_changes( 
					( self.layers[layer].get_all_regular_weight_changes( channel ) + 
						weight_gradients[ layer ][ channel ] ), channel )
				print( "\n\n\n\n\n")
				


			lastOneAccessed -= 3
			layer -= 1


	def train( self, inputs, target_outputs, epochs, batch_size, order ):
		for epoch in range( epochs ):
			new_epoch = True
			if order != ORDER_4:
				if (epoch == 0) & (order == ORDER_0):
					permutation = get_random_permutation( inputs )
					#print( "permutation: " + str( permutation ) )
					inputs = randomise( inputs, permutation )
					target = randomise( target_outputs, permutation )
					inputs = np.reshape( inputs, ( len( inputs ), np.size( inputs[0] ) ) )
					target = np.reshape( target, ( len( target ), np.size( target[0] ) ) )
					#print( "inputs: " + str( inputs ) )
					#print( "target: " + str( target ) )
				elif ( epoch == 0 ):
					inputs = np.reshape( inputs, ( len( inputs ), np.size( inputs[0] ) ) )
					target = np.reshape( target_outputs, ( len( target_outputs ), np.size( target_outputs[0] ) ) )

			
			batch_order = get_batch_order( order, epoch, inputs, batch_size )

			
 			for index in range( len(batch_order) ):	
				batch_input = inputs[batch_order[index]*batch_size: 
						batch_order[index]*batch_size + batch_size ]
				batch_target = target[batch_order[index]*batch_size:
						batch_order[index]*batch_size + batch_size ]
				self.hypothesis( batch_input.flatten() )
				self.back_propagate( batch_input.flatten(), batch_target )
		
			for l in range( len( self.layers ) ):
				for channel in range( self.input_channels ):
					self.layers[l].set_all_regular_weights(
						(self.layers[l].get_regular_weights( channel ) - 
							learning_rate*( (1.0/len(batch_order))*self.layers[l].get_all_regular_weight_changes( channel ) +
							 self.weight_penalty*self.layers[l].get_regular_weights( channel ) ) ), channel )
					#print( "regular weight changes before: " + str( self.layers[l].get_all_regular_weight_changes( channel ) ) )
				
				#self.layers[l].set_all_bias_weights( self.layers[l].get_all_bias_weights() -
				#	learning_rate*( (1.0/len(batch_order))*self.layers[l].get_all_bias_weight_changes() ) )

				# Regular and bias weight change reset to 0, to enable more learning based on subsequent epoch. 
					self.layers[l].set_all_regular_weight_changes( np.zeros( ( self.layers[ l ].get_regular_weights( channel ).shape[0], 
								self.layers[ l ].get_regular_weights( channel ).shape[1] ) ), channel )
				#self.layers[l].set_all_bias_weight_changes( np.zeros( ( self.layers[l].get_bias_weight_dimensionality()[1], 
				#	self.layers[l].get_bias_weight_dimensionality()[0] ) ) )
			
				
			

# Convolutional N.N. B.-P. Cases:
# Max pooling 
# Min pooling
# Mean pooling

# Cases to consider:
# image width == odd; image height == odd; lrc == even; 1, 2, 3
# image width == odd; image height == even; lrc == even; 1, 2, 3
# image width == even; image height == odd; lrc == even; 1, 2, 3
# image width == even; image height == even; lrc == even; 1, 2, 3
# image width == odd; image width == odd; lrc == odd; 1, 2, 3
# image width == odd; image height == even; lrc == odd; 1, 2, 3
# image width == even; image height == odd; lrc == odd; 1, 2, 3


# constraints:
# if input_depth is of odd dimension, 
# then so, too, must receptive_field
# step cannot exceed recetive_field
# receptive_field_cannot exceed input_depth


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

# similar constraints apply to compute_convolutional_layer_breadth()
# new method
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

# applicable to contiguous regions
def is_dimensionality_compatible( image_size, receptive_field_radii ):
	compatible = True
	running_dimensionality = float( image_size )
	index = 0

	while ( compatible & (index < len( receptive_field_radii ) )):
		if ( running_dimensionality / receptive_field_radii[index] >= 1 ):
			running_dimensionality = np.ceil( running_dimensionality / receptive_field_radii[index] )
			index += 1
		else:
			return False
	return True

# applicable to non-contiguous regions; i.e., over-lapping regions
def universal_is_dimensionality_compatible( image_size, receptive_field_radii, step ):
	compatible = True
	running_dimensionality = float( image_size )
	index = 0

	while ( compatible & ( index < len( receptive_field_radii ) ) ):
		if ( running_dimensionality / receptive_field_radii[ index ] >= 1 ):
			running_dimensionality = np.ceil( running_dimensionality - receptive_field_radii[ index ] + step )
			index += 1
		else:
			return False
	return True

def is_odd(num):
    return num & 0x1

if __name__ == "__main__":
	
	#data = load_data()
	#inputs = data[0][0][0]
	inputs = np.zeros( (2,108) )
	inputs[0] = np.arange( 0, 1.08, 0.01 )
	inputs[1] = np.arange( 0, 1.08, 0.01 )
	input_channels = 3
	input_layer_dimensionality = [6,6]
	pooling_types = [1,1]
	learning_rate = 0.1
	regular_weight_init_range = [0.1,0.2] 
	bias_weight_init_range = [0.1,0.2]
	identification = "Convolutional Network"
	order = [[0,0],[1,0],[0,1]]
	receptive_fields = [3,3,2]

 	network_test = ConvolutionalNetwork( input_channels, 
		input_layer_dimensionality, 
		pooling_types, learning_rate, regular_weight_init_range, 
		bias_weight_init_range, identification, order, receptive_fields, 0.01 )
 	network_test.build_network()
 
 	#network_test.hypothesis( inputs[0] )
 	target_output = np.matrix([[1.0],[0.0]])
 	#train( self, np.asmatrix( inputs ), target_outputs, epochs, batch_size, order )
 	network_test.train( np.asmatrix( inputs ), target_output, 1, 1, 0 )
 	print( "ancillary storage: " + str( len( network_test.ancillary_storage ) ) )
 	#network_test.back_propagate( inputs, target_output )
 	#print( "inputs: " + str( inputs ) )
 	"""
 	print "data: " + str( data[0][0][0] )

 	# format for reading in data:
 	# Access the first normalised training image: data[0][0][0]
 	# Access the corresponding training image label: data[0][1][0]
 	"""
 	"""

 	print( "Techincally Layer 0:" )
 	for channel in range( input_channels ):
 		print( "regular weight gradients: " + str( 
 			network_test.layers[0].get_all_regular_weight_changes( channel ) ) )
 		print( "stored regular weight changes: " + str( 
 			network_test.weight_gradients[0][channel] ) )
	"""
 	"""
 	print( "layers[0].activations( 0 ): " + str( network_test.layers[0].get_all_node_activations( 0 ) ) )
 	print( "layers[0].activations( 1 ): " + str( network_test.layers[0].get_all_node_activations( 1 ) ) )
  	print( "layers[0].activations( 2 ): " + str( network_test.layers[0].get_all_node_activations( 2 ) ) )

  	print( "layers[1].activations( 0 ): " + str( network_test.layers[1].get_all_node_activations( 0 ) ) )
  	print( "layers[1].activations( 1 ): " + str( network_test.layers[1].get_all_node_activations( 1 ) ) )
  	print( "layers[1].activations( 2 ): " + str( network_test.layers[1].get_all_node_activations( 2 ) ) )

  	print( "layers[2].activations( 0 ): " + str( network_test.layers[2].get_all_node_activations( 0 ) ) )
  	print( "layers[2].activations( 1 ): " + str( network_test.layers[2].get_all_node_activations( 1 ) ) )
  	print( "layers[2].activations( 2 ): " + str( network_test.layers[2].get_all_node_activations( 2 ) ) )
 	"""
