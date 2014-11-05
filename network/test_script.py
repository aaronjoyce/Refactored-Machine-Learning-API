current_layer_width = 8
current_layer_height = 8
next_layer_width = 5
next_layer_height = 5
rfs = 4


for row in range( current_layer_height ):
	for col in range( current_layer_width ):
		#rfs = self.layers[layer+1].get_rfs()
		#next_layer_height = self.layers[layer+1].get_height()
		#next_layer_width = self.layers[layer+1].get_width()
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

