class Layer(object):
	"""docstring for Layer"""
	def __init__(self, arg):
		super(Layer, self).__init__()
		self.arg = arg
		

class InputLayer( Layer ):
	"""docstring for InputLayer"""
	def __init__( self, arg ):
		super(InputLayer, self ).init__()
		self.arg

class ConvolutionalLayer( Layer ):
	"""docstring for ConvolutionalLayer"""
	def __init__(self, arg):
		super(ConvolutionalLayer, self).__init__()
		self.arg = arg

class Pooling( Layer ):
	"""docstring for Pooling"""
	def __init__(self, arg):
		super(Pooling, self).__init__()
		self.arg = arg	
		
class MinPooling(Pooling):
	"""docstring for MinPooling"""
	def __init__(self, arg):
		super(MinPooling, self).__init__()
		self.arg = arg

class MaxPooling(Pooling):
	"""docstring for MaxPooling"""
	def __init__(self, arg):
		super(MaxPooling, self).__init__()
		self.arg = arg
		
class MeanPooling(Pooling):
	"""docstring for MeanPooling"""
	def __init__(self, arg):
		super(MeanPooling, self).__init__()
		self.arg = arg

class OutputLayer(Layer):
	"""docstring for OutputLayer"""
	def __init__(self, arg):
		super(OutputLayer, self).__init__()
		self.arg = arg

class FullyConnectedLayer(Layer):
	"""docstring for FullyConnectedLayer"""
	def __init__(self, arg):
		super(FullyConnectedLayer, self).__init__()
		self.arg = arg
		