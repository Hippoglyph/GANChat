import tensorflow as tf

class Generator():
	def __init__(self, embedding, sequence_length):
		self.sequence_length = sequence_length
		self.embedding = embedding
		self.encoder_units = 4
		self.buildGraph()

	def buildGraph(self):
		with tf.name_scope("generator"):
			self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_sequence")

			self.embedded_input = self.embedding.getEmbedding(self.input_x)

			with tf.name_scope("encoder"):
				self.encoder_state, self.encoder_final_state = tf.nn.dynamic_rnn(
															tf.nn.rnn_cell.GRUCell(self.encoder_units, name="encoder_cell", kernel_initializer = tf.contrib.layers.xavier_initializer()),
															self.embedded_input,
															dtype=tf.float32)


			with tf.name_scope("decoder"):

				decoder_cell = LSTMCell(decoder_hidden_units)

				W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
				b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

				
				def loop_fn_initial():
					initial_elements_finished = (0 >= decoder_lengths)
					initial_input = eos_step_embedded
					initial_cell_state = encoder_final_state
					initial_cell_output = None
					initial_loop_state = None
					return (initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state)

				def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

					def get_next_input():
						output_logits = tf.add(tf.matmul(previous_output, W), b)
						prediction = tf.argmax(output_logits, axis=1)
						next_input = tf.nn.embedding_lookup(embeddings, prediction)
						return next_input
					
					elements_finished = (time >= decoder_lengths)

					finished = tf.reduce_all(elements_finished)
					next_input_ = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
					state = previous_state
					output = previous_output
					loop_state = None

					return (elements_finished, next_input_, state, output, loop_state)

				def loop_fn(time, previous_output, previous_state, previous_loop_state):
					if previous_state is None:
						return loop_fn_initial()
					else:
						return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

				decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
				decoder_outputs = decoder_outputs_ta.stack()

				decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
				decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
				decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
				decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
				
				decoder_prediction = tf.argmax(decoder_logits, 2)