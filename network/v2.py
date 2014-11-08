import numpy as np


Y_AXIS_MEAN = 0
ORDER_0 = 0
ORDER_1 = 1
ORDER_2 = 2 
ORDER_3 = 3 
ORDER_4 = 4

class EdgeGroup: 
	def __init__( self, edge_init_range, x_dimension, y_dimension ):
		self.range = edge_init_range
		self.x_dimension = x_dimension
		self.y_dimension = y_dimension

		self.edges = None
		self.gradients = None
	
	def initialise_random_edges( self ):
		self.edges = np.matrix( np.random.uniform( 
			self.range[0], self.range[1], self.x_dimension * self.y_dimension 
			).reshape( self.x_dimension, self.y_dimension ) )

	def initialise( self ):
		self.edges = np.matrix( np.random.uniform( 
			self.range[0], self.range[1], self.x_dimension * self.y_dimension ) )


	def get_edges( self ):
		return self.edges

	def get_gradients( self ):
		return self.gradients

	# return a 1-dimensional, 2-element array
	def get_dimensionality( self ):
		return [self.x_dimension, self.y_dimension]

	# returns a Boolean, indicating whether the edges
	# have been updated
	def set_edges( self, edges ):
		# checks whether the argument passed is of 
		# the correct dimensionality
		self.edges = edges

	def set_gradients( self, gradients ):
		if ( len( gradients ) == self.y_dimension & 
			len( gradients[0] ) == self.x_dimension ):
			self.gradients = gradients
			return True
		return False



class Network:
	def __init__( self, input_dimensionality, 
		hidden_layer_dimensionality, learning_rate, 
		bias_node, regular_weight_init_range, 
		bias_weight_init_range, weight_penalty, output_dimensionality = None,
		identification = None,  
		connectivity = None ):

		self.input_dimensionality = input_dimensionality
		self.hidden_layer_dimensionality = hidden_layer_dimensionality
		self.learning_rate = learning_rate
		self.bias_node = bias_node
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range
		self.output_dimensionality = output_dimensionality
		self.id = identification
		self.connectivity = connectivity
		self.weight_penalty = weight_penalty

		# A one-dimensional array of Layer objects.
		self.layers = []

	def build_network( self ):
		# i.e., if desired network is ! an auto-encoder
		print "build_network() invoked" 
		for l in range( len( self.hidden_layer_dimensionality ) + 1 ):
			# '0' should be replaced by a constant, like 'FIRST_LAYER'
			if l == 0:
				self.layers.append( Layer( str( l ), self.input_dimensionality, 
					regular_weight_init_range, self.hidden_layer_dimensionality[l][1], 
					self.hidden_layer_dimensionality[l][0], self.bias_weight_init_range ) )
				self.layers[l].build_layer();
			elif l == len( self.hidden_layer_dimensionality ):
				self.layers.append( Layer( str( l ), self.hidden_layer_dimensionality[l-1], 
					regular_weight_init_range, self.output_dimensionality[1], 
					self.output_dimensionality[0], self.bias_weight_init_range ) )
				self.layers[l].build_layer();
			else:
				if self.output_dimensionality != None:
					self.layers.append( Layer( str( l ), self.hidden_layer_dimensionality[l-1], 
						self.regular_weight_init_range, self.hidden_layer_dimensionality[l][1], 
						self.hidden_layer_dimensionality[l][0], self.bias_weight_init_range ) )
					self.layers[l].build_layer();
		print "network built"
	def get_layers( self ):
		return self.layers
		

	def train( self, inputs, target_outputs, epochs, batch_size, order ):
		for epoch in range( epochs ):
			for layer in range( len( self.layers ) ):
				print( "epoch: " + str( epoch ) )
				print( "layer: " + str( layer ) )
				print( "self.layers[layer].get_all_regular_weights(): " + str( 
					self.layers[layer].get_all_regular_weights() ) )
			new_epoch = True
			if order != ORDER_4:
				if epoch == 0 & order == ORDER_0:
					permutation = get_random_permutation( inputs )
					inputs = randomise( inputs, permutation )
					target = randomise( target_outputs, permutation )
					inputs = np.reshape( inputs, ( len( inputs ), np.size( inputs[0] ) ) )
					target = np.reshape( target, ( len( target), np.size( target[0] ) ) )

			batch_order = get_batch_order( order, epoch, inputs, batch_size )
			#print "batch_order: " + str( batch_order )
 			for index in range( len(batch_order) ):	
				batch_input = inputs[batch_order[index]*batch_size: 
						batch_order[index]*batch_size + batch_size ]
				batch_target = target[batch_order[index]*batch_size:
						batch_order[index]*batch_size + batch_size ]
				self.back_propagate( batch_input, batch_target )
			for l in range( len( self.layers ) ):
				self.layers[l].set_all_regular_weights(
					(self.layers[l].get_all_regular_weights() - 
						self.learning_rate*( (1.0/len(batch_order))*self.layers[l].get_regular_weight_changes() +
						 self.weight_penalty*self.layers[l].get_all_regular_weights() ) ) )
				self.layers[l].set_all_bias_weights( self.layers[l].get_all_bias_weights() -
					self.learning_rate*( (1.0/len(batch_order))*self.layers[l].get_all_bias_weight_changes() ) )

				# Regular and bias weight change reset to 0, to enable more learning based on subsequent epoch. 
				self.layers[l].set_all_regular_weight_changes( np.zeros( ( self.layers[l].get_regular_weight_dimensionality()[1], 
					self.layers[l].get_regular_weight_dimensionality()[0] ) ) )
				self.layers[l].set_all_bias_weight_changes( np.zeros( ( self.layers[l].get_bias_weight_dimensionality()[1], 
					self.layers[l].get_bias_weight_dimensionality()[0] ) ) )
		
		
	def hypothesis( self, inputs ):
		for layer in range( len( self.layers ) ):
			if layer == 0:
				self.layers[layer].update_all_node_activations( htan( np.dot( 
					np.append( np.array( inputs ), np.ones( (len(inputs),1) ), 
						axis=1), self.layers[layer].get_all_weights() ) ) )
				self.layers[layer].set_average_node_activation( np.mean( self.layers[layer].get_all_node_activations() ) )

			else:
				self.layers[layer].update_all_node_activations( htan( np.dot( 
					np.append( np.array( self.layers[layer-1].get_all_node_activations() ),
					 np.ones( (len( self.layers[layer-1].get_all_node_activations() ),1)), axis=1 ),
					  self.layers[layer].get_all_weights() )))

		return self.layers[len( self.layers ) - 1 ].get_all_node_activations() 
			

	# 'hypothesis' and 'target' must be one-dimensional - re-shaping can be 
	# performed upon the returned data structure. 
	def compute_output_error( self, hypothesis, target ):
		# CHANGE MADE: original incorrect implementation: 
		return np.multiply( -( target - hypothesis  ), hypothesis*(1-hypothesis) )

	# Parameters:
	# 'activations' and 'errors' must be column vectors (can be tensors of col. vectors, too)
	# Assumed form of weights:
	# [[w1_1,w1_2,w1_3,...,w1_n],
	#  [w2_1,w2_2,w2_3,...,w2_n],
	#			...
	#  [wm_1,wm_2,wm_3,...,wm_n]].
	# Returns a column vector (possibly a tensor of
	# column vectors). 
	# Q. Could 'activations' and 'errors' be an "array" of vectors?
	# Q. Will it be necessary to performa transformation upon 'weights'?
	def compute_hidden_error( self, activations, errors, weights ):
		# problem likely pertains to 'activations'
		return np.transpose( np.multiply( np.dot( np.transpose( weights ), np.transpose(errors) ),
			np.transpose( np.multiply( activations, (1-activations) ) ) ) )

	# Both parameters ('activations' and 'errors') must be column 
	# vectors - and not tensors of column vectors
	def compute_regular_weight_gradients( self, activations, errors ):
		temp = []
		for batch_instance in range( np.size(errors) ):
			temp.append(np.transpose( np.multiply( np.take( errors, 
				[batch_instance], 1 ), activations ) ) )

			#return np.mean( temp, 0, None, None )
		#print "what's returned: " + str( np.transpose( np.transpose(temp)[0]  )  )
		return np.transpose( np.transpose( temp )[0] )

	# Parameter: Column vector
	def compute_bias_weight_gradients( self, errors ):
		return np.mean( errors, 1 )
		
	def back_propagate( self, inputs, target_output ):
		errors = [0]*( len( self.layers ) )
		weight_gradients = [0]*( len( self.layers ) )
		bias_weight_gradients = [0]*( len( self.layers ) )
		errors[ len(self.layers) - 1 ] = self.compute_output_error( 
			self.hypothesis( inputs ), target_output )
		layer = len( self.layers ) - 1 
		while ( layer >= 0 ):
			if ( layer <= len( self.layers ) - 2 ):

				errors[layer] = self.compute_hidden_error( self.layers[layer].get_all_node_activations(), 
					errors[layer+1], self.layers[layer+1].get_all_regular_weights() ) 
			if ( layer != 0 ):
				weight_gradients[layer] = self.compute_regular_weight_gradients( np.mean( 
					self.layers[layer-1].get_all_node_activations(), Y_AXIS_MEAN ), 
				np.matrix(np.mean( errors[layer], Y_AXIS_MEAN )) )
				bias_weight_gradients[layer] = np.transpose( np.mean( 
					errors[layer], Y_AXIS_MEAN ) )

			else:
				weight_gradients[layer] = self.compute_regular_weight_gradients( np.mean( 
					inputs, Y_AXIS_MEAN ), np.matrix(np.mean( errors[layer], Y_AXIS_MEAN)) )
				bias_weight_gradients[layer] = np.transpose( np.mean( errors[layer], Y_AXIS_MEAN ) )
			self.layers[layer].set_all_regular_weight_changes( ( 
				self.layers[layer].get_regular_weight_changes() + weight_gradients[layer]) )
			self.layers[layer].set_all_bias_weight_changes( 
				self.layers[layer].get_all_bias_weight_changes() +
				 np.transpose( np.matrix(bias_weight_gradients[layer]) ) )
			layer -= 1

		print "hypothesis: " + str( self.hypothesis( inputs ) )
		print "target: " + str( target_output )
		print( "weight gradients: " + str( weight_gradients ) )
		  		
			

# 'input_dimensionality': a 2-element, one-dimensional array. First
# element refers to the x-dimension; the second to the y-dimension. 

class Layer:
	def __init__( self, identity, input_dimensionality, 
		regular_weight_init_range, depth, breadth = 1, 
		bias_weight_init_range = None, biases = True ):
		# layer dimesnionality; i.e., depth and breadth

		self.depth = depth # no. of rows
		self.breadth = breadth # no. of columns
		self.biases = biases
		self.identity = identity
		self.input_dimensionality = input_dimensionality #[x_length, y_length]
		self.regular_weight_init_range = regular_weight_init_range
		self.bias_weight_init_range = bias_weight_init_range 

		# By default, we begin with a 1-dimensional 
		# data structure. Its internal structure
		# may be augmented within the 'build_layer()'
		# function.

		self.nodes = np.empty(( 1, self.depth * self.breadth  ), dtype=object)
		self.bias_node = None
		self.average_node_activation = None


	def get_average_node_activation( self ):
		return self.average_node_activation

	def set_average_node_activation( self, average_activation ):
		self.average_node_activation = average_activation

	def build_layer( self ):
		# where 'breadth' exceeds 1, 
		# a two-dimensional node structure 
		# is required
		#re-write
		# structure acquires the form of an attribute, rather than something
		# more explicit
		for node in range( self.depth * self.breadth ):
			self.nodes[0][ node ] = Node( self.input_dimensionality[1], 
					node % self.breadth, node / self.breadth, self.input_dimensionality[0], self.regular_weight_init_range )
					
		self.bias_node = BiasNode( None, node % self.breadth, node / self.breadth, 1.0 )

	def get_all_nodes( self ):
		return self.nodes 

	# Returns the Node object at the position x, y, where 
	# x and y denote the Node's x- and y-coordinates, respectively. 
	# @param: A one-dimensional, two-element array. The array's first 
	# element contains the x-coordinate; the second the y-coordinate.  
	# @return: A single Node object. 
	# @raises: NodeActivationIndexException. This exception is raised
	# if the indices specified do not correspond to valid x- and y-coordinates
	# within the Layer object. 
	def get_node( self, index ):
		# internal representation of 'self.nodes': 
		# a column vector, where the first element of 
		# 'index' denotes the x-coordinate; the second the y-coordinate
		if index[0] < self.breadth & index[1] < self.depth:
			return np.take( self.nodes, [index[1]*self.breadth + index[0]], 1)
		else:
			raise NodeActivationIndexException( "A Node does not exist at the " + 
				"x- and y-coordinate combination specified." )

	# @param 'nodes': A one- or two-dimensional matrix/Numpy array
	# of witdh equal to the total number of Nodes of
	# the Layer instance. The width of 'nodes' is equal to the 
	# depth of the Layer times (multiplied by) the breadth 
	# of the Layer. The depth of the layer (the number of row)
	# is equal to the number of activations in the active 
	# batch of input. 
	def update_all_node_activations( self, nodes ):
		for node in range( self.depth * self.breadth ):
			self.nodes[0][ node ].set_node( np.take( nodes, [node], 1 ) )



	# Returns the activation(s) associated with a Node object
	# at the specified index. 
	# @param 'index': A one-dimensional, two-element array of ints. 
	# The first element of the array denotes the x-coordinate of the Node;
	# the second element of the array denotes the y-coordinate. 
	# @raises NodeActivationIndexException. 
	def get_node_activation( self, index ):
		if ((index[0] < self.breadth) & (index[1] < self.depth)):
			return self.nodes[ 0 ][index[1]*self.breadth + index[0] ].get_node()
		else:
			raise NodeActivationIndexException( "A Node does not exist at the x-" + 
				" and y-coordinate combination specified." )

	# Return a one- or two-dimensional array/matrix (depending on the number of
	# of inputs in the active batch instance). 
	# The width of the returned array/matrix is equal to 
	# the breadth times (multiplied by) the depth of the Layer. 
	# The depth of the returned matrix is dependent upon the number
	# of inputs in the active batch instance. 
	def get_all_node_activations( self ):
		temp = []
		for node in range( self.depth * self.breadth ):
			temp.append( self.nodes[0][node].get_node() )
		return np.transpose( temp )[0]
		
	# Updates the Real-valued activation value associated
	# with the Node at position x, y, where 'x' 
	# and 'y' refer to the Node's x- and y-coordinates, respectively. 
	# @param 'index': A one-dimensional, two-element array of ints. 
	# The first element of the array denotes the x-coordinate of the Node;
	# the second element of the array denotes the y-coordinate. 
	# @param 'activation': A Real-valued number. 
	# @return a Boolean value, indicating whether the Node at the specified position
	# has had its activation value updated. If 'true' is returned, the Node's activation
	# has been successfully updated. If 'false' is returned, the Node's activation
	# has not been updated. 

	def update_node_activation( self, index, activation ):
		if ( index[0] < self.breadth & index[1] < self.depth ):
			self.nodes[0][ index[1]*self.breadth + index[0] ].set_node( node )
			return True
		else:
			raise NodeActivationIndexException( "A Node does not exist at the x-" + 
				" and y-coordinate combination specified." )
		return False

	# Returns the depth of the Layer.
	# By 'depth' is meant the distance from the vertical distance from 
	# the Layer's highest to lowest node. 
	# @returns: A non-zero Integer or None. It returns a 
	# non-zero Integer when a depth value is specified as a 
	# parameter of Layer during instantiation. Where 'None' 
	# was passed at the time of instantiation of Layer, it returns
	# None. 
	def get_depth( self ):
		return self.depth

	# Returns the breadth/width of the Layer, where 
	# the returned value refers to the distance between 
	# the left-most and right-most nodes at any non-zero row
	# of the Layer. 
	# @ returns: A non-zero Integer or None. It return a non-zero
	# Integer when a breadth value is specified as a parameter of 
	# Layer during instantiation. When 'None' is passed as 
	# a parameter at the time of instantiation of Layer, the function 
	# returns None.
	def get_breadth( self ):
		return self.breadth
	
	# Return a single BiasNode instance associated with the 
	# Layer instance. 
	def get_bias_nodes( self ):
		return self.bias_nodes
		
	# Returns the activation (a Real number) (generally, though not invariantly, 1.0)
	# associated with the BiasNode instance of the Layer instance. 
	def get_bias_node_activation( self ):
		return self.bias_node.get_activation()

	# return a one-dimensional array of vectors, where
	# each vector denotes the regular weights associated with 
	# a particular node. Where the nodes are of the form:
	# nodes = [[n1,n2,n3]
	#  [n4,n5,n6]
	#  [n7,n8,n9]],
	# the corresponding array is of the form:
	# all regular weights = [ [w,w,w,w,w,...,w], 
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w],
	#   [w,w,w,w,w,...,w] ], where len( all regular weights ) == 9 -- the same flattened length of 'nodes'.
	# Each element of the one-dimensional array is a flattened representation 
	# of an internally-represented two-dimensional matrix of weights associated 
	# with each node.
	# It is imperative to note that each vector must be of type 'matrix'.
	# This function returns a two-dimensional array of weights, where each row-vector of 
	# array denotes the weights connecting a specific node in layer 'l + 1' to all nodes (or inputs)
	# in layer 'l'. 
	def get_all_regular_weights( self ):
		temp = np.empty( ( self.depth*self.breadth, self.input_dimensionality[0]*self.input_dimensionality[1] ) )
		for node in range( self.depth * self.breadth ):
			temp[node] = self.nodes[0][node].get_regular_weights().flatten()
		return temp
	
	# Returns a one- or two-dimensional matrix. The width/breadth 
	# of the matrix is equal to the total number of Nodes in the next lower
	# layer, where the current layer may be represented by 'l' and 
	# the next lower layer by 'l-1'. The depth of the returned 
	# matrix is equal to the total number of Nodes in the current layer, 'l', plus 1. 
	# The the final row of Real numbers is equal to all of the bias weights connecting
	# the Nodes in layer 'l' to the BiasNode in layer l-1. 
	def get_all_weights( self ):
		return np.transpose( np.append( self.get_all_regular_weights(), self.get_all_bias_weights(), 1 ) )
		
	# Returns a one- or two-dimensional matrix. The width/breadth 
	# of the matrix is equal to the total number of Nodes in the next lower 
	# layer, where the current layer may be represented by 'l' and
	# the next lower layer by 'l-1'. The depth of the returned 
	# matrix is equal to the total number of Nodes in the current layer, 'l'. 	
	def get_regular_weight_changes( self ):
		temp = []
		#print "self.nodes[0][0]: " + str( self.nodes[0][0].get_regular_weight_changes() )
		for node in range( self.depth * self.breadth ):
			temp.append( np.reshape( self.nodes[0][ node ].get_regular_weight_changes().flatten(), 
				( np.size( self.nodes[0][node].get_regular_weight_changes() ), 1 ) ) )
		
		return np.transpose(np.transpose( temp )[0])
	
	# Returns a column vector of Real numbers. 
	# The Real-value number at row zero denotes
	# the weight on the Edge between the first Node 
	# instance (x=0,y=0), while the Real-value number
	# at the row equal to the length of the column vector - 1
	# denotes the weight associated with Edge connecting
	# the Node at coordinates max(x),max(y) to the BiasNode in the next lower
	# layer. 
	def get_all_bias_weights( self ):
		temp = np.empty( (self.depth * self.breadth, 1 ) )

		for node in range( self.depth * self.breadth ):
			temp[node] = self.nodes[0][node].get_bias_weight()
		return temp

	# Returns a column vector of Real numbers. 
	# The Real-value number at row zero denotes
	# the weight change associated with the Edge between 
	# the first Node instance (x=0,y=0), while the 
	# Real-value number at the row equal to the length of 
	# the column vector - 1 denotes the weight change associated 
	# with Edge connecting the Node at coordinates max(x),may(y)
	# to the BiasNode in the next lower layer. 
	def get_all_bias_weight_changes( self ):
		temp = []
		for node in range( self.depth * self.breadth ):
			temp.append( self.nodes[0][node].get_bias_weight_changes() )
		return np.transpose( np.transpose( temp )[0] )
		
	"""
	Function specification:
	@param: 'weights'
	Each vector denotes the regular weights associated with 
	a particular node. Where the nodes are of the form:
	nodes = [[n1,n2,n3]
			[n4,n5,n6]
			[n7,n8,n9]],
	the corresponding array is of the form:
	all regular weights = [ [w,w,w,w,w,...,w], 
		[w,w,w,w,w,...,w],
	  	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w],
	   	[w,w,w,w,w,...,w] ], where len( all regular weights ) == 9 -- the same flattened length of 'nodes'
	"""
	def set_all_regular_weights( self, weights ):
		for node in range( self.depth * self.breadth ):
			self.nodes[0][node].set_regular_weights( weights[ node ] )

	# @param 'weights': A column vector of Real-value bias weights.
	# The Real-value number at row zero denotes
	# the bias weight associated with the Edge between 
	# the first Node instance (x=0,y=0) and the singular BiasNode
	# in the next lower layer, while the 
	# Real-value number at the row equal to the length of 
	# the column vector - 1 denotes the bias weight associated 
	# with Edge connecting the Node at coordinates max(x),may(y)
	# to the same BiasNode in the next lower layer. 
	def set_all_bias_weights( self, weights ):
		for node in range( self.depth * self.breadth ):
			self.nodes[0][node].set_bias_weight( weights[ node ] )
	
	# @param: 'weight_changes': A one- or two-dimensional 
	# Each vector denotes the regular weights associated with 
	# a particular node. Where the nodes are of the form:
	# nodes = [[n1,n2,n3]
	#		  [n4,n5,n6]
	#		  [n7,n8,n9]],
	# the corresponding array is of the form:
	# all regular weights = [ [wc,wc,wc,wc,wc,...,wc], 
	#	    [wc,wc,wc,wc,wc,...,wc],
	#  	    [wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc],
	#   	[wc,wc,wc,wc,wc,...,wc] ], where len( all regular weights ) == 9 -- the same flattened length of 'nodes'.	
	def set_all_regular_weight_changes( self, weight_changes ):
		for node in range( self.depth * self.breadth ):
			self.nodes[0][node].set_regular_weight_changes( weight_changes[node] )


	# @param 'weight_changes': A column vector of Real-value bias weights changes.
	# The Real-value number at row zero denotes
	# the bias weight change associated with the Edge between 
	# the first Node instance (x=0,y=0) and the singular BiasNode
	# in the next lower layer, while the 
	# Real-value number at the row equal to the length of 
	# the column vector - 1 denotes the bias weight change associated 
	# with Edge connecting the Node at coordinates max(x),may(y)
	# to the same BiasNode in the next lower layer. 
	def set_all_bias_weight_changes( self, weight_changes ):
		for node in range( self.depth * self.breadth ):
			self.nodes[0][node].set_bias_weight_changes( np.take( weight_changes, [node], 0 ) )

	# Returns a one-dimensional, two-element array.
	# The first element (at index 0) is equal to the
	# y-dimensionality (number of rows) of the regular weights associated 
	# with the Layer instance; the second element (at index 1)
	# is equal to the x-dimensionality (number of columns) of the regular weights
	# associated with the Layer instance.
	def get_regular_weight_dimensionality( self ):
		return [ self.input_dimensionality[0] * self.input_dimensionality[1], 
			self.depth * self.breadth ]

	# Returns a one-dimensional, two-element array. 
	# The first element (at index 0) is equal to the
	# y-dimensionality (numbers of rows) of the regular weights
	# associated with the Layer instance; the second element (at index 1)
	# is equal to the x-dimensionality (number of columns) of the bias weights
	# associated with the Layer instance. 
	def get_bias_weight_dimensionality( self ):
		return [ 1, self.depth * self.breadth ]

class BiasNode:
	def __init__( self, layer, x_coord, y_coord, activation ):
		self.layer = layer
		self.x_coord = x_coord
		self.y_coord = y_coord
		self.activation = activation
		
	# Returns a singular Real-value number denoting
	# the activation associated with the BiasNode instance. 
	# It is typically, though not invariably, 1.0. 
	def get_activation( self ):
		return self.activation
	
	# Sets the singular activation associated with the
	# BiasNode instance. 
	# @param 'update_activations': A singular Real-value number. 
	def set_activation( self, updated_activation ):
		self.activation = updated_activation

		

# Attributes of a Layer: 
# - Edges
# - Nodes



class Node:
	def __init__( self, height_of_previous_layer, x, y,
		width_of_previous_layer, weight_init_range, voxel = None ):
		# an array of EdgeGroup object?
		self.weights =  EdgeGroup( 
			weight_init_range, 
			1, width_of_previous_layer * height_of_previous_layer )
		self.weight_changes = EdgeGroup( 
			[0,0], width_of_previous_layer, height_of_previous_layer )
		self.weights.initialise_random_edges()
		self.weight_changes.initialise_random_edges()

		# self.weights should reference an array of Edge objects,
		# and each Edge object should have properties; e.g., 
		# direction; gradient; x and y dimensions; and weight. 
		self.bias_weight = EdgeGroup( 
			weight_init_range, 1, 1 )
		self.bias_weight_changes = EdgeGroup( 
			[0,0], 1, 1 )
		self.bias_weight.initialise_random_edges()
		self.bias_weight_changes.initialise_random_edges()
		# self.bias_weight should, too, reference an Edge
		# object and this should have properties; e.g.,
		# weight; gradient; and x and y dimensions. 

		self.activation = None
		self.error = None
		self.x = x
		self.y = y
		self.voxel = voxel

	def set_node( self, activation ):
		#print "activation within set_node(): " + str( activation )
		self.activation = activation

	def get_node( self ):
		return self.activation

	def set_error( self, error ):
		self.error = error

	def get_error( self ):
		return self.error

	def set_regular_weights( self, weights ):
		self.weights.set_edges( weights )

	def get_regular_weights( self ):
		return self.weights.get_edges()

	def set_bias_weight( self, weight ):
		self.bias_weight.set_edges( weight )

	def get_bias_weight( self ):
		return self.bias_weight.get_edges()
		
	def get_regular_weight_changes( self ):
		return self.weight_changes.get_edges() 
	
	def set_regular_weight_changes( self, weight_changes ):
		self.weight_changes.set_edges( weight_changes )
		
	def get_bias_weight_changes( self ):
		return self.bias_weight_changes.get_edges()
		
	def set_bias_weight_changes( self, weight_changes ):
		self.bias_weight_changes.set_edges( weight_changes )
	
	def get_x( self ):
		return self.x

	def get_y( self ):
		return self.y

	def get_voxel( self ):
		return self.voxel


# 'helper' functions
def sigmoid( x ):
	return 1/(1+np.exp(-x))
	
def htan( x ):
	return (np.exp(2*x)-1)/(np.exp(2*x)+1)

# input: an array of matrices
def randomise( input, permutation ):
	temp = []
	for unit in range( len( permutation ) ):
		temp.append( input[ permutation[ unit ] ] ) 
	return temp


def get_batch_order( order, epoch, input, batch_size ):
	if order == ORDER_1 | ( order == ORDER_2 & epoch == 0 ):
		temp = np.arange( int( np.ceil( len(input)/float( batch_size ) ) ) ) 
		np.random.shuffle( temp )
		return temp.tolist()
	return np.arange( int( np.ceil( len( input )/float(batch_size) ) ) ).tolist()

def get_random_permutation( input ):
	temp = np.arange( len( input ) )
	np.random.shuffle( temp )
	return temp

class SparseAutoencoderNetwork( Network ):
	def __init__( self, input_dimensionality, 
		hidden_layer_dimensionality, learning_rate, 
		bias_node, regular_weight_init_range, 
		bias_weight_init_range, weight_penalty, 
		sparsity_penalty, sparsity_constraint, 
		output_dimensionality = None, identification = None, 
		connectivity = None ):

		Network.__init__( self, input_dimensionality, 
			hidden_layer_dimensionality, learning_rate, 
			bias_node, regular_weight_init_range, 
			bias_weight_init_range, weight_penalty,
			output_dimensionality, identification, connectivity )

		self.sparsity_penalty = sparsity_penalty
		self.sparsity_constraint = sparsity_constraint

	
	#@Overrides( Network )
	def hypothesis( self, inputs ):
		for layer in range( len( self.layers ) ):
			if layer == 0:
				self.layers[layer].update_all_node_activations( htan( np.dot( 
					np.append( np.array( inputs ), np.ones( (len(inputs),1) ), 
						axis=1), self.layers[layer].get_all_weights() ) ) )
				self.layers[layer].set_average_node_activation( np.mean( self.layers[layer].get_all_node_activations() ) )

			else:
				self.layers[layer].update_all_node_activations( htan( np.dot( 
					np.append( np.array( self.layers[layer-1].get_all_node_activations() ),
					 np.ones( (len( self.layers[layer-1].get_all_node_activations() ),1)), axis=1 ),
					  self.layers[layer].get_all_weights() )))
			self.layers[layer].set_average_node_activation( np.mean( self.layers[layer].get_all_node_activations() ) )

		return self.layers[len( self.layers ) - 1 ].get_all_node_activations() 

	# + B*( -(p_object/average_p) + (1-p_object)/(1-p_average) ) )
	#@Overrides( Network )
	def compute_hidden_error( self, activations, errors, weights, average_activation ):
		# problem likely pertains to 'activations'
		return np.transpose( np.multiply( ( np.dot( np.transpose( weights ), np.transpose(errors) ) +
			self.sparsity_penalty*( -(self.sparsity_constraint/average_activation) + 
				(1-self.sparsity_constraint)/(1-average_activation) ) ),
			np.transpose( np.multiply( activations, (1-activations) ) ) ) )

	#@Overrides( Network )
	def compute_output_error( self, hypothesis, target, average_activation ):
		# CHANGE MADE: original incorrect implementation: 
		return np.multiply( (-( target - hypothesis  )+self.sparsity_penalty*( -(self.sparsity_constraint/average_activation) + 
				(1-self.sparsity_constraint)/(1-average_activation) )), hypothesis*(1-hypothesis) )
	#@Overrides( Network )
	def back_propagate( self, inputs, target_output ):
		errors = [0]*( len( self.layers ) )
		weight_gradients = [0]*( len( self.layers ) )
		bias_weight_gradients = [0]*( len( self.layers ) )
		errors[ len(self.layers) - 1 ] = self.compute_output_error( 
			self.hypothesis( inputs ), target_output, self.layers[ len( self.layers)-1 ].get_average_node_activation() )
		layer = len( self.layers ) - 1 
		while ( layer >= 0 ):
			if ( layer <= len( self.layers ) - 2 ):
				errors[layer] = self.compute_hidden_error( self.layers[layer].get_all_node_activations(), 
					errors[layer+1], self.layers[layer+1].get_all_regular_weights(), self.layers[layer].get_average_node_activation() ) 
			if ( layer != 0 ):
				weight_gradients[layer] = self.compute_regular_weight_gradients( np.mean( 
					self.layers[layer-1].get_all_node_activations(), Y_AXIS_MEAN ), 
				np.matrix(np.mean( errors[layer], Y_AXIS_MEAN )) )
				bias_weight_gradients[layer] = np.transpose( np.mean( 
					errors[layer], Y_AXIS_MEAN ) )

			else:
				weight_gradients[layer] = self.compute_regular_weight_gradients( np.mean( 
					inputs, Y_AXIS_MEAN ), np.matrix(np.mean( errors[layer], Y_AXIS_MEAN)) )
				bias_weight_gradients[layer] = np.transpose( np.mean( errors[layer], Y_AXIS_MEAN ) )
			self.layers[layer].set_all_regular_weight_changes( ( 
				self.layers[layer].get_regular_weight_changes() + weight_gradients[layer]) )
			self.layers[layer].set_all_bias_weight_changes( 
				self.layers[layer].get_all_bias_weight_changes() +
				 np.transpose( np.matrix(bias_weight_gradients[layer]) ) )
			layer -= 1


# Attributes of a Node:
# - Activation
# - Error

class NodeActivationIndexException( Exception ):
	def __init__( self, value ):
		self.value = value
	def __str__( self ):
		return repr( self.value )


if __name__ == "__main__":
			input_dimensionality = [3,1]
			hidden_layer_dimensionality = [[5,1],[4,1]]
			learning_rate = 3.0
			bias_node = 1.0
			regular_weight_init_range = [0.0,0.1]
			bias_weight_init_range = [0.0,0.1]
			weight_penalty = 0.01
			output_dimensionality = [3,1]
			sparsity_penalty = 0.01
			sparsity_constraint = -1.0
			identification = "Test Network"
			#target_output = np.matrix([[1.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0],[1.0,1.0,1.0,1.0]])
			target_output = np.matrix([[1.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,1.0]])
		
			network_object = Network( input_dimensionality, 
				hidden_layer_dimensionality, learning_rate, bias_node, 
				regular_weight_init_range, bias_weight_init_range, 
				weight_penalty, output_dimensionality, identification )
			network_object.build_network()
			
			"""
			sparse_autoencoder = SparseAutoencoderNetwork( 
				input_dimensionality, 
				hidden_layer_dimensionality, learning_rate, 
				bias_node, regular_weight_init_range, 
				bias_weight_init_range, weight_penalty, 
				sparsity_penalty, sparsity_constraint, output_dimensionality, identification )
			sparse_autoencoder.build_network()
			"""
		

			# test as you program!
			#inputs = np.matrix([[1.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0],[1.0,1.0,1.0,1.0]])
			inputs = np.matrix([[1.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,1.0]])


			# 'inputs' should be of its "natural" form;
			# 'target_output' should be in flat form 

			#hypothesis = network_object.hypothesis( inputs )
			
			network_object.train( inputs, target_output, 3, 3, 0 )
			
			"""
			#sparse_autoencoder.train( inputs, target_output, 2, 3, 0 )
			print "average node activation in Layer 0: " + str( network_object.layers[0].get_average_node_activation() )
			print "average node activation in Layer 1: " + str( network_object.layers[1].get_average_node_activation() )
			print "get_all_nodes: " + str( network_object.layers[0].get_all_nodes() )
			print "activation: " + str( network_object.layers[0].nodes[0][0].get_node() )
			

			# the architecture of CNNs is designed to take advantage of 2-dimensional inputs
			
			for i in range( hidden_layer_dimensionality[0][0]*hidden_layer_dimensionality[0][1] ):
				print "x: " + str( network_object.layers[0].nodes[0][i].get_x() )
				print "y: " + str( network_object.layers[0].nodes[0][i] )
 
			print "regular weights (l1): " + str( np.shape( network_object.layers[1].get_all_regular_weights() ) )
			print "regular weight changes (l1): " + str( np.shape( network_object.layers[1].get_regular_weight_changes() ) )
			print "bias weights (l1): " + str( np.shape( network_object.layers[1].get_all_bias_weights() ) )
			print "bias weight changes (l1): " + str( np.shape( network_object.layers[1].get_all_bias_weight_changes() ) )

			print "regular weights (l0): " + str( np.shape( network_object.layers[0].get_all_regular_weights() ) )
			print "regular weight changes (l0): " + str( np.shape( network_object.layers[0].get_regular_weight_changes() ) )
			print "bias weights (l0): " + str( np.shape( network_object.layers[0].get_all_bias_weights() ) )
			print "bias weight changes (l0): " + str( np.shape( network_object.layers[0].get_all_bias_weight_changes() ) )
			"""
			

"""
Development notes:

- Weight gradients are temporal
"""


	# High-level description of what it is the function does. 
	# Requires
	# Effects
	


#Your Gadget Insurance Validation Certificate number: - BIGIIAJ/20141

# Attributes of a Node:
# - Activation
# - Error