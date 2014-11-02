import numpy as np 

"""
def extract_rows( row_step, lrc_size ):
	for row in range( input_width - lrc_size + 1 ):
		for col in range( input_width - lrc_size + 1 ):
			indices = acquire_indices( row, col, row_step, lrc_size, input_indices )
			print( "Groups: " + str( inputs[indices] ) )
"""

def extract_rows( row_step, lrc_size, row, col, input_indices, inputs ):
	indices = acquire_indices( row, col, row_step, lrc_size, input_indices )
	return inputs[indices]


def acquire_indices( row, col, row_step, lrc_size, input_indices ):
	return input_indices[ row : row + lrc_size : 
				row_step, col : col + lrc_size ].flatten()



if __name__ == "__main__":
	input_width = 5
	input_height = 5
	instances_per_batch = 3
	lrc_size = 4
	ROW_STEP = 1


	input_indices = np.arange( input_width * input_height ).reshape( input_height, input_width )
	inputs = np.empty((input_width * input_height)*instances_per_batch);
	inputs = inputs.reshape(input_width * input_height, instances_per_batch);
	for i in range( input_width * input_height ):
		inputs[i].fill( i + 1 )

	print extract_rows( ROW_STEP, 3, 0, 0, input_indices, inputs )