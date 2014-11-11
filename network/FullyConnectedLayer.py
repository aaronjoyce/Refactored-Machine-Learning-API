public class FullyConnectedLayer( Layer ):
	def __init__(self, identity, input_width, input_height, 
		layer_width, layer_height, channels, rfs, bias_nodes, regular_weight_init_range, 
		bias_weight_init_range = None, biases = True ):
		Layer.__init__( identity, input_width, input_height, 
			layer_width, layer_height, channels, rfs, bias_nodes, 
			regular_weight_init_range, bias_weight_init_range )

	def assemble( self ):
		for channel in range( self.channels ):
			self.regular_weights.append( EdgeGroup( self.regular_weight_init_range, 
				self.layer_width * self.layer_height * input_width * input_height, 
				sefl.DEFAULT_Y_DIMENSION ) )
			self.regular_weight_changes.append( EdgeGroup( [0.0,0.0], 
				self.layer_width * self.layer_height * input_width * input_height, 
				sefl.DEFAULT_Y_DIMENSION ) )
			self.regular_weight_changes[ len( self.regular_weight_changes )  - 1 ].initialise()
			if ( self.biases ):
				self.bias_weights.append( EdgeGroup( self.bias_weight_init_range, 
					self.DEFAULT_X_DIMENSION, self.layer_width * self.layer_height ) )
				self.bias_weights[ len( self.bias_weights ) - 1 ].initialise()
				self.bias_weight_changes.append( EdgeGroup( self.bias_weight_init_range, 
					self.DEFAULT_X_DIMENSION, self.layer_width * self.layer_height ) )
				self.bias_weight_changes[ len( self.bias_weight_changes ) - 1 ].initialise()