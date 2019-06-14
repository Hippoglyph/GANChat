import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper

class decoderHelper(Helper):
	def __init__(self, keep_length, sequence_length, embedding, reply_seq_embedding, start_token_embedding, W, b, _batch_size):
		self.keep_length = keep_length
		self.sequence_length = sequence_length
		self.embedding = embedding
		self.W = W
		self.b = b
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
		logits = tf.add(tf.matmul(state.h, self.W), self.b)
		prob = tf.nn.softmax(logits)
		return tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self._batch_size])

	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished= tf.greater_equal(time+1, self.sequence_length)
		if self.keep_length <= 0:
			next_token = self.embedding.getEmbedding(sample_ids)
		else:
			next_token = tf.cond(tf.logical_or(tf.greater_equal(time+1, self.keep_length), finished), lambda: self.embedding.getEmbedding(sample_ids), lambda: self.reply_embedding[:,time,:])
		return finished, next_token, state

class Generator():
	def __init__(self, embedding, sequence_length, start_token, vocab_size, learning_rate, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 2000
		self.noiseSize = 256
		self.noiseStd = 1.0
		self.decoder_units = self.encoder_units + self.noiseSize
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.scope_name = "generator"
		self.buildGraph()

	def generate(self, sess, post, noise=True):
		if(noise):
			output = sess.run(
					self.seqences[0],
					{self.post_seq: post})
		else:
			noise = np.zeros((self.batch_size, self.noiseSize))
			output = sess.run(
					self.seqences[0],
					{self.post_seq: post, self.noise: noise})
		return output

	def pretrain(self, sess, post, reply):
		noise = np.zeros((self.batch_size, self.noiseSize))
		loss_summary, loss, _ = sess.run(
				[self.pretrain_summary, self.pretrain_loss, self.pretrain_update],
				{self.post_seq: post,self.reply_seq: reply, self.noise: noise})
		return loss_summary, loss

	def rolloutStep(self, sess, post, reply, keepIndex):
		return sess.run(
			self.seqences[keepIndex],
			{self.post_seq: post, self.reply_seq: reply})

	def calculateReward(self, sess, post, reply, tokenSampleRate, discriminator):
		rewards = np.zeros((self.batch_size, self.sequence_length))
		for keepNumber in range(1,self.sequence_length):
			for _ in range(tokenSampleRate):
				sampleReply = self.rolloutStep(sess, post, reply, keepNumber)
				rewards[:,keepNumber-1] += discriminator.evaluate(sess, post, sampleReply)
			rewards[:,-1] = discriminator.evaluate(sess, post, reply) * tokenSampleRate
		return rewards / tokenSampleRate

	def train(self, sess, post, reply, rewards):
		loss_summary, loss, _ = sess.run(
				[self.train_summary, self.loss, self.update],
				{self.post_seq: post, self.reply_seq: reply, self.rewards: rewards})
		return loss_summary, loss

	def buildInputGraph(self):
		with tf.variable_scope("inputs"):
			self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
			self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
			self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length], name="rewards")
			self.noise = tf.placeholder_with_default(tf.random_normal([self.batch_size, self.noiseSize], 0.0, self.noiseStd), shape=[self.batch_size, self.noiseSize], name="noise")
		
			#self.batch_size = tf.shape(self.post_seq)[0]
			self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			self.embedded_start_token = self.embedding.getEmbedding(self.start_token)

			self.embedded_post = self.embedding.getEmbedding(self.post_seq)

			self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)

	def buildTrainingGraph(self, sequence, logits):
		with tf.variable_scope("training"):
			self.generatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
			#for r in self.generatorVariables:
			#	print(r.name)

			with tf.variable_scope("pretrain"):
				#logits = tf.log(tf.clip_by_value(self.seqences_logits[self.sequence_length], 1e-8,1-1e-8))
				self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reply_seq, logits=logits))
				pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.generatorVariables), 5.0)
				self.pretrain_update = pretrain_optimizer.apply_gradients(zip(self.pretrain_grad, self.generatorVariables))
				self.pretrain_summary = tf.summary.scalar("generator_pretrain_loss", self.pretrain_loss)

			with tf.variable_scope("RL-learning"):
				genProb = tf.nn.softmax(logits)
				genLogProb = tf.log(tf.clip_by_value(genProb, 1e-8,1-1e-8))
				self.loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(sequence, self.vocab_size) * genLogProb, -1) * self.rewards)
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.generatorVariables), 5.0)
				self.update = optimizer.apply_gradients(zip(self.grad, self.generatorVariables))
				self.train_summary = tf.summary.scalar("generator_loss", self.loss)

	def buildChoModel(self):
		with tf.variable_scope("encoder"):

			encoder_RNN = rnn.LSTMCell(self.encoder_units)

			_, self.encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(encoder_RNN, self.embedded_post, dtype=tf.float32)

		with tf.variable_scope("decoder"):

			decoder_RNN = rnn.LSTMCell(self.encoder_units)
			decoder_RNN_W0 = tf.Variable(tf.random_normal([self.encoder_units, self.vocab_size], stddev=0.1), name="LSTM_output_decoder_1_W0")
			decoder_RNN_B0 = tf.Variable(tf.random_normal([self.vocab_size], stddev=0.1), name="LSTM_output_decoder_1_B0")

			with tf.variable_scope("noise"):
				h0 = self.encoder_final_hidden_memory_tuple[1]
				c0 = self.encoder_final_hidden_memory_tuple[0]
				h0 = tf.concat([h0, self.noise], 1)
				c0 = tf.concat([c0, self.noise], 1)
				decoderH0 = rnn.LSTMStateTuple(c0,h0)

			with tf.variable_scope("generate"):

				seqences = []
				seqences_logits = []
				for iteration_number in range(self.sequence_length+1):
					decoder = seq2seq.BasicDecoder(
						cell=decoder_RNN,
						helper=decoderHelper(iteration_number, self.sequence_length, self.embedding, self.embedded_reply, self.embedded_start_token, decoder_RNN_W0, decoder_RNN_B0, self.batch_size),
						initial_state=self.encoder_final_hidden_memory_tuple
						)

					final_outputs, _, _ = seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=self.sequence_length)

					seqences.append(tf.reshape(final_outputs.sample_id, [self.batch_size, self.sequence_length])) # batch_size x sequence_length
					logits = tf.add(tf.tensordot(final_outputs.rnn_output, decoder_RNN_W0, axes=[[2], [0]]), decoder_RNN_B0[None, None,:])
					seqences_logits.append(tf.reshape(logits,[self.batch_size, self.sequence_length, self.vocab_size])) # batch_size x sequence_length x vocab_size

		return seqences, seqences_logits

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			self.buildInputGraph()

			self.seqences, self.seqences_logits = self.buildChoModel()

			self.buildTrainingGraph(self.seqences[self.sequence_length], self.seqences_logits[self.sequence_length])
			