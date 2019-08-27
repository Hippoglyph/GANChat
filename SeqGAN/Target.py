import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper

class DecoderHelper(Helper):
	def __init__(self, sequence_length, embedding, start_token_embedding, _batch_size):
		self.sequence_length = sequence_length
		self.embedding = embedding
		self._batch_size = _batch_size
		self.start_embedding = start_token_embedding

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return tf.int32

	def initialize(self, name=None):
		finished = tf.tile([False], [self._batch_size])
		return finished, self.start_embedding

	def sample(self, time, outputs, state, name=None):
		prob = tf.nn.softmax(outputs)
		return tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self._batch_size])

	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished= tf.greater_equal(time+1, self.sequence_length)
		next_token = self.embedding.getEmbedding(sample_ids)
		return finished, next_token, state

class EvalHelper(Helper):
	def __init__(self, sequence_length, embedding, reply_seq_embedding, start_token_embedding, _batch_size):
		self.sequence_length = sequence_length
		self.embedding = embedding
		self._batch_size = _batch_size
		self.start_embedding = start_token_embedding
		self.reply_embedding = reply_seq_embedding

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def sample_ids_shape(self):
		return tf.TensorShape([])

	@property
	def sample_ids_dtype(self):
		return tf.int32

	def initialize(self, name=None):
		finished = tf.tile([False], [self._batch_size])
		return finished, self.start_embedding

	def sample(self, time, outputs, state, name=None):
		return tf.constant(0, dtype=tf.int32, shape=[self._batch_size])

	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished= tf.greater_equal(time+1, self.sequence_length)
		next_token = self.reply_embedding[:,time,:]
		return finished, next_token, state

class Target():
	def __init__(self, embedding, sequence_length, start_token, vocab_size, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.units = 32
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.scope_name = "target"
		self.buildGraph()

	def generate(self, sess):
		output = sess.run(
				self.seqence)
		return output

	def getProbability(self, sess, reply):
		output = sess.run(
				self.score,
				{self.reply_seq: reply})
		return output

	def calculateScore(self, sess, generator, total_iteration):
		nll = []
		for _ in range(total_iteration):
			sequence = generator.generate(sess)
			nll.append(self.getProbability(sess, sequence))
		return np.mean(nll)

	def buildInputGraph(self):
		with tf.variable_scope("inputs"):
			self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
			#self.batch_size = tf.shape(self.post_seq)[0]
			self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			self.embedded_start_token = self.embedding.getEmbedding(self.start_token)
			self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)

	def buildModel(self):
		with tf.variable_scope("decoder"):

			decoder_RNN = rnn.LSTMCell(self.units, initializer=tf.initializers.random_normal(0,1.0))
			decoder_RNN_projection = tf.layers.Dense(units=self.vocab_size, use_bias=True, kernel_initializer=tf.initializers.random_normal(0,1.0))

			with tf.variable_scope("generate"):
				decoder = seq2seq.BasicDecoder(
					cell=decoder_RNN,
					helper=DecoderHelper(self.sequence_length, self.embedding, self.embedded_start_token, self.batch_size),
					initial_state=decoder_RNN.zero_state(self.batch_size, dtype=tf.float32),
					output_layer=decoder_RNN_projection
					)

				final_outputs, _, _ = seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=self.sequence_length)

				seqence = tf.reshape(final_outputs.sample_id, [self.batch_size, self.sequence_length]) # batch_size x sequence_length

			with tf.variable_scope("evaluate"):
				eval_decoder = seq2seq.BasicDecoder(
					cell=decoder_RNN,
					helper=EvalHelper(self.sequence_length, self.embedding, self.embedded_reply, self.embedded_start_token, self.batch_size),
					initial_state=decoder_RNN.zero_state(self.batch_size, dtype=tf.float32),
					output_layer=decoder_RNN_projection
					)

				eval_final_outputs, _, _ = seq2seq.dynamic_decode(decoder=eval_decoder, maximum_iterations=self.sequence_length)

				probs = tf.nn.softmax(tf.reshape(eval_final_outputs.rnn_output,[self.batch_size, self.sequence_length, self.vocab_size])) # batch_size x sequence_length x vocab_size

		return seqence, probs

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			self.buildInputGraph()

			self.seqence, self.probs = self.buildModel()

			self.score = -tf.reduce_sum(tf.one_hot(self.reply_seq, self.vocab_size) * tf.log(self.probs))/(self.sequence_length * self.batch_size)