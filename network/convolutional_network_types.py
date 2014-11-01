class InputType:
	def __init__( self ):
		pass

class OutputType:
	def __init__( self ):
		pass

class ConvolutionalType:
	def __init__( self ):
		pass

class FullyConnectedType( OutputType ):
	def __init__( self ):
		pass

class PoolingType:
	def __init__( self ):
		pass

class MinPoolingType( PoolingType ):
	def __init__( self ):
		pass

class MaxPoolingType( PoolingType ):
	def __init__( self ):
		pass

class MeanPoolingType( PoolingType ):
	def __init__( self ):
		pass

class NoSubtype( InputType, ConvolutionalType ):
	def __init__( self ):
		pass