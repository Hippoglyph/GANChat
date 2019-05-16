import tensorflow as tf

class Generator():
	def __init__(self, embedding, sequence_length):
		self.sequence_length = sequence_length
		self.embedding = embedding

		self.buildGraph()

	def buildGraph(self):
		with tf.name_scope("generator"):
			self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_sequence")

			self.embedded_input = self.embedding.getEmbedding(self.input_x)

			'''
			cell = tf.nn.rnn_cell.LSTMCell(hidden, state_is_tuple=True)
			outputs, state = tf.nn.dynamic_rnn(cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
            '''