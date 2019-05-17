import tensorflow as tf

class Embedding():
	def __init__(self, vocab_size, embedding_size):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.buildGraph()

	def buildGraph(self):
		#with tf.device('/cpu:0'), tf.name_scope("embedding"):
		with tf.name_scope("embedding"):
			self.EM = tf.Variable(
				tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
				name="EM")

	def getEmbedding(self, inputIds):
		#with tf.device('/cpu:0'):
			return tf.nn.embedding_lookup(self.EM, inputIds)

	def getParams(self):
		return [self.EM]