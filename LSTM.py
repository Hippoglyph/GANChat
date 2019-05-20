import tensorflow as tf

class LSTM():
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

	def init_matrix(self, shape):
		return tf.random_normal(shape, stddev=0.1)

	def init_vector(self, shape):
		return tf.zeros(shape)

	def create_recurrent_unit(self, params):
		# Weights and Bias for input and hidden tensor
		self.Wi = tf.Variable(self.init_matrix([self.input_size, self.hidden_size]))
		self.Ui = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bi = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wf = tf.Variable(self.init_matrix([self.input_size, self.hidden_size]))
		self.Uf = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bf = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wog = tf.Variable(self.init_matrix([self.input_size, self.hidden_size]))
		self.Uog = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bog = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wc = tf.Variable(self.init_matrix([self.input_size, self.hidden_size]))
		self.Uc = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bc = tf.Variable(self.init_matrix([self.hidden_size]))
		params.extend([
			self.Wi, self.Ui, self.bi,
			self.Wf, self.Uf, self.bf,
			self.Wog, self.Uog, self.bog,
			self.Wc, self.Uc, self.bc])

		def unit(x, hidden_memory_tm1):
			previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

			# Input Gate
			i = tf.sigmoid(
				tf.matmul(x, self.Wi) +
				tf.matmul(previous_hidden_state, self.Ui) + self.bi
			)

			# Forget Gate
			f = tf.sigmoid(
				tf.matmul(x, self.Wf) +
				tf.matmul(previous_hidden_state, self.Uf) + self.bf
			)

			# Output Gate
			o = tf.sigmoid(
				tf.matmul(x, self.Wog) +
				tf.matmul(previous_hidden_state, self.Uog) + self.bog
			)

			# New Memory Cell
			c_ = tf.nn.tanh(
				tf.matmul(x, self.Wc) +
				tf.matmul(previous_hidden_state, self.Uc) + self.bc
			)

			# Final Memory cell
			c = f * c_prev + i * c_

			# Current Hidden state
			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return unit

	def create_output_unit(self, params):
		self.Wo = tf.Variable(self.init_matrix([self.hidden_size, self.input_size]))
		self.bo = tf.Variable(self.init_matrix([self.input_size]))
		params.extend([self.Wo, self.bo])

		def unit(hidden_memory_tuple):
			hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
			# hidden_state : batch x hidden_dim
			logits = tf.matmul(hidden_state, self.Wo) + self.bo
			# output = tf.nn.softmax(logits)
			return logits

		return unit