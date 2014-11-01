import numpy as np 

if __name__ == "__main__":
	input_width = 110
	input_height = 110
	instances_per_batch = 50
	lrc_size = 4
	ROW_STEP = 1

	input_indices = np.arange( input_width * input_height ).reshape( input_height, input_width )
	inputs = np.empty((input_width * input_height)*instances_per_batch);
	inputs = inputs.reshape(input_width * input_height, instances_per_batch);
	for i in range( input_width * input_height ):
		inputs[i].fill( i + 1 )
	
	for row in range( input_width - lrc_size + 1 ):
		for col in range( input_width - lrc_size + 1 ):
			indices = input_indices[ row : row + lrc_size : 
				ROW_STEP, col : col + lrc_size ].flatten()
			print( "Groups: " + str( inputs[indices] ) )







