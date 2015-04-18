import pyspark

# don't do this
class SearchFunctions(object):
	def __init__(self, query):
		self.query = query

	def isMatch(self, s):
		return query in s

	def getMatchesFunctionReference(self, rdd):
		# Problem: References all of "self" in "self.isMatch"
		return rdd.filter(self.isMatch)

	def getMatchesMemberReference(self, rdd):
		# Problem: References all of "self" in "self.query"
		return rdd.filter(lambda x: self.query in x)

class WordFunction(object):
	...
	def getMatchesNoReference(self, rdd):
		# Safe: Extract only the filed we need into a local variable
		query = self.query
		return rdd.filter(lambda x: query in x)