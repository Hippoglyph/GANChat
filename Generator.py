import tensorflow as tf
from LSTM import LSTM_recurrent_unit, LSTM_output_unit
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Generator():
	def __init__(self, embedding, sequence_length, start_token):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 4
		self.decoder_units = self.encoder_units #Add noice here
		self.params = []
		self.buildGraph()

	def buildGraph(self):
		with tf.name_scope("generator"):
			self.input_x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="input_sequence")
			self.batch_size = tf.shape(self.input_x)[0]
			self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			
			self.params.extend(self.embedding.getParams())

			self.embedded_input = self.embedding.getEmbedding(self.input_x)
			#with tf.device("/cpu:0"):
			self.embedded_input_transposed = tf.transpose(self.embedded_input, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

			with tf.name_scope("encoder"):

				embedded_input_lt = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
				embedded_input_lt = embedded_input_lt.unstack(self.embedded_input_transposed)
				
				self.encoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units,self.params)

				self.eh0 = tf.zeros([self.batch_size, self.encoder_units])
				self.eh0 = tf.stack([self.eh0, self.eh0])

				def encoder_loop(i, x_t, h_tm1):
					h_t = self.encoder_LSTM(x_t, h_tm1)
					return i + 1, embedded_input_lt.read(i), h_t

				_,_, self.encoder_final_hidden_memory_tuple = control_flow_ops.while_loop(
					cond=lambda i, _1, _2: i < self.sequence_length,
					body=encoder_loop,
					loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), self.eh0))

				self.encoder_final_state,_ = tf.unstack(self.encoder_final_hidden_memory_tuple)
			


	
	
	
				