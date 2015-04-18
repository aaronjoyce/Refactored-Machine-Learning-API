class Worker:
	def __init__( start_index, chunk_size, interval_updates ):
		self.start_index = start_index
		self.chunk_size = chunk_size
		self.interval_updates = interval_updates
		self.last_interval = start_index
		self.active = False
		self.ti

	def get_start_index( self ):
		return self.start_index

	def get_chunk_size( self ):
		return self.chunk_size

	def is_active( self ):
		return self.active

	def set_status( self, status ):
		self.active = status

	def names_processed( self ):
		return self.last_interval




