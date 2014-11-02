import numpy as np

class EdgeGroup: 
	def __init__( self, edge_init_range, x_dimension, y_dimension ):
		self.edge_range = edge_init_range
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
			self.range[0], self.range[1], self.x_dimension * self.y_dimension 
			).reshape( self.x_dimension, self.y_dimension ) )


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