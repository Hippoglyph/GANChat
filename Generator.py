import tensorflow as tf
from LSTM import LSTM_recurrent_unit, LSTM_output_unit
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator():
	def __init__(self, embedding, sequence_length, start_token, vocab_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 4
		self.decoder_units = self.encoder_units #Add noice here
		self.vocab_size = vocab_size
		self.params = []
		self.buildGraph()

	def generate(self, sess, inputs):
		output = sess.run(
				[self.gen_seq],
				{self.input_seq: inputs})
		return output


	def buildGraph(self):
		with tf.name_scope("generator"):
			with tf.name_scope("inputs"):
				self.input_seq = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_sequence")
				self.target_seq = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="target_sequence")
			
				self.batch_size = tf.shape(self.input_seq)[0]
				self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			
				self.params.extend(self.embedding.getParams())

				self.embedded_input = self.embedding.getEmbedding(self.input_seq)
				#with tf.device("/cpu:0"):
				self.embedded_input_transposed = tf.transpose(self.embedded_input, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

				self.embedded_pretrain_target = self.embedding.getEmbedding(self.target_seq)
				#with tf.device("/cpu:0"):
				self.embedded_pretrain_target_transposed = tf.transpose(self.embedded_pretrain_target, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

			with tf.name_scope("encoder"):

				embedded_input_lt = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
				embedded_input_lt = embedded_input_lt.unstack(self.embedded_input_transposed)
				
				self.encoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units, "encoder_1", self.params)

				self.eh0 = tf.zeros([self.batch_size, self.encoder_units])
				self.eh0 = tf.stack([self.eh0, self.eh0])

				def encoder_loop(i, x_t, h_tm1):
					h_t = self.encoder_LSTM(x_t, h_tm1)
					return i + 1, embedded_input_lt.read(i), h_t

				_,_, self.encoder_final_hidden_memory_tuple = control_flow_ops.while_loop(
					cond=lambda i, _1, _2: i < self.sequence_length,
					body=encoder_loop,
					loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), self.eh0))

			with tf.name_scope("decoder"):

				self.decoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units, "decoder_1",self.params)
				self.decoder_LSTM_output = LSTM_output_unit(self.vocab_size,self.encoder_units, "decoder_1", self.params)

				with tf.name_scope("generate"):
					self.gen_seq = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)

					def gen_loop(i, x_t, h_tm1, gen_seq):
						h_t = self.decoder_LSTM(x_t, h_tm1)
						o_t = self.decoder_LSTM_output(h_t) # batch_size x vocab_size
						log_prob = tf.log(tf.nn.softmax(o_t))
						next_token = tf.reshape(tf.multinomial(log_prob, 1, output_dtype=tf.int32), [self.batch_size])
						x_tp1 = self.embedding.getEmbedding(next_token)
						gen_seq = gen_seq.write(i, next_token)
						return i + 1, x_tp1, h_t, gen_seq

					_,_,_,self.gen_seq = control_flow_ops.while_loop(
						cond=lambda i, _1, _2, _3: i < self.sequence_length,
						body=gen_loop,
						loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), self.encoder_final_hidden_memory_tuple, self.gen_seq))

					self.gen_seq = tf.transpose(self.gen_seq.stack(), perm=[1,0]) # batch_size x sequence_length

				with tf.name_scope("pretrain"):
					self.pretrain_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)

					embedded_target_lt = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
					embedded_target_lt = embedded_target_lt.unstack(self.embedded_pretrain_target_transposed)

					def decoder_pretrain_loop(i, x_t, h_tm1, pretrain_predictions):
						h_t = self.decoder_LSTM(x_t, h_tm1)
						o_t = self.decoder_LSTM_output(h_t)
						pretrain_predictions = pretrain_predictions.write(i, tf.nn.softmax(o_t))
						x_tp1 = embedded_target_lt.read(i)
						return i + 1, x_tp1, h_t, pretrain_predictions

					_,_,_,self.pretrain_predictions = control_flow_ops.while_loop(
						cond=lambda i, _1, _2, _3: i < self.sequence_length,
						body=decoder_pretrain_loop,
						loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), self.encoder_final_hidden_memory_tuple, self.pretrain_predictions))

					self.pretrain_predictions = tf.transpose(self.pretrain_predictions.stack(), perm=[1,0,2]) # batch_size x sequence_length x vocab_size

					self.pretrain_loss = -tf.reduce_sum( #Cross entropy loss
			            tf.one_hot(tf.reshape(self.target_seq, [-1]), self.vocab_size, 1.0, 0.0) * # (batch_size x sequence_length) x vocab_size
			            	tf.log(
			                	tf.clip_by_value(tf.reshape(self.pretrain_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
				            )
			        	) / (tf.cast(self.sequence_length * self.batch_size,dtype=tf.float32))


			


	
	
	
				