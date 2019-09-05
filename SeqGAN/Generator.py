import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class DecoderHelper(Helper):
	def __init__(self, keep_length, sequence_length, embedding, reply_seq, start_token_embedding, _batch_size):
		self.keep_length = keep_length
		self.sequence_length = sequence_length
		self.embedding = embedding
		self._batch_size = _batch_size
		self.start_embedding = start_token_embedding
		self.reply = reply_seq

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
		sample_ids = tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self._batch_size])
		if self.keep_length <= 0:
			next_token = sample_ids
		else:
			next_token = tf.cond(tf.greater(time+1, self.keep_length), lambda: sample_ids, lambda: self.reply[:,time])
		return next_token

	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished = tf.greater_equal(time+1, self.sequence_length)
		next_token = self.embedding.getEmbedding(sample_ids)
		return finished, next_token, state

class Generator():
	def __init__(self, embedding, sequence_length, start_token, vocab_size, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.units = 32
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = 1e-2
		self.scope_name = "generator"
		self.buildGraph()

	def generate(self, sess):
		output = sess.run(
				self.seqence)
		return output

	def pretrain(self, sess, reply):
		loss_summary, loss, _ = sess.run(
				[self.pretrain_summary, self.pretrain_loss, self.pretrain_update],
				{self.reply_seq: reply})
		return loss_summary, loss

	def rolloutStep(self, sess, reply, keepIndex):
		return sess.run(
			self.seqence,
			{self.reply_seq: reply, self.keepNumber: keepIndex})

	def calculateReward(self, sess, reply, tokenSampleRate, discriminator):
		rewards = np.zeros((self.batch_size, self.sequence_length))
		for keepNumber in range(1,self.sequence_length):
			for _ in range(tokenSampleRate):
				sampleReply = self.rolloutStep(sess, reply, keepNumber)
				rewards[:,keepNumber-1] += discriminator.evaluate(sess, sampleReply)
		rewards[:,-1] = discriminator.evaluate(sess, reply) * tokenSampleRate
		return rewards / tokenSampleRate

	def train(self, sess, reply, rewards):
		loss_summary, loss, _ = sess.run(
				[self.train_summary, self.loss, self.update],
				{self.reply_seq: reply, self.rewards: rewards})
		return loss_summary, loss

	def buildInputGraph(self):
		with tf.variable_scope("inputs"):
			self.reply_seq = tf.placeholder_with_default(tf.zeros([self.batch_size, self.sequence_length], dtype=tf.int32), shape=[self.batch_size, self.sequence_length], name="reply_sequence")
			#self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
			self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length], name="rewards")
			self.keepNumber = tf.placeholder_with_default(0, shape=())
		
			#self.batch_size = tf.shape(self.post_seq)[0]
			self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			self.embedded_start_token = self.embedding.getEmbedding(self.start_token)

			self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)

			self.ta_reply = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
			self.ta_reply = self.ta_reply.unstack(tf.transpose(self.reply_seq, perm=[1, 0])) #seq_length x batch

			self.ta_emb_reply = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
			self.ta_emb_reply = self.ta_emb_reply.unstack(tf.transpose(self.embedded_reply, perm=[1, 0, 2])) #seq_length x batch x embedding

	def buildTrainingGraph(self, logits):
		with tf.variable_scope("training"):
			self.generatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
			#for r in self.generatorVariables:
				#print(r.name)
				#print(r.shape)

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
				self.loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.reply_seq, self.vocab_size) * genLogProb, -1) * self.rewards)
				#self.loss = tf.reduce_mean(tf.reduce_sum(tf.one_hot(sequence, self.vocab_size) * genLogProb, -1) * self.rewards)# ?? WTF
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.generatorVariables), 5.0)
				self.update = optimizer.apply_gradients(zip(self.grad, self.generatorVariables))
				self.train_summary = tf.summary.scalar("generator_loss", self.loss)

	def buildModel(self):

		with tf.variable_scope("decoder"):
			std = 0.1
			self.cell = rnn.LSTMCell(self.units, initializer=tf.initializers.random_normal(0,0.1))
			self.W = tf.Variable(tf.random_normal([self.units, self.vocab_size], stddev=std), name="W")
			self.b = tf.Variable(tf.random_normal([self.vocab_size], stddev=std), name="b")
			#decoder_RNN_projection = tf.layers.Dense(units=self.vocab_size, use_bias=True, kernel_initializer=tf.initializers.random_normal(0,0.1))

			'''
			with tf.variable_scope("generate"):
				seqences = []
				seqences_logits = []
				for iteration_number in range(self.sequence_length+1):
					decoder = seq2seq.BasicDecoder(
						cell=decoder_RNN,
						helper=DecoderHelper(iteration_number, self.sequence_length, self.embedding, self.reply_seq, self.embedded_start_token, self.batch_size),
						initial_state=decoder_RNN.zero_state(self.batch_size, dtype=tf.float32),
						output_layer=decoder_RNN_projection
						)

					final_outputs, _, _ = seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=self.sequence_length)

					seqences.append(tf.reshape(final_outputs.sample_id, [self.batch_size, self.sequence_length])) # batch_size x sequence_length
					seqences_logits.append(tf.reshape(final_outputs.rnn_output,[self.batch_size, self.sequence_length, self.vocab_size])) # batch_size x sequence_length x vocab_size
			'''
			with tf.variable_scope("generate"):

				sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
				seqence_logits = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

				def loop_keep(time, inputs, cell_state, sequence, keepNumber):
					_, next_cell_state = self.cell(inputs, cell_state)
					next_inputs = self.ta_emb_reply.read(time)
					sequence = sequence.write(time, self.ta_reply.read(time))
					return time + 1, next_inputs, next_cell_state, sequence, keepNumber

				def loop_gen(time, inputs, cell_state, sequence):
					outputs, next_cell_state = self.cell(inputs, cell_state)
					logits = tf.add(tf.matmul(outputs, self.W), self.b)
					prob = tf.nn.softmax(logits)
					sample_ids = tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self.batch_size])
					sequence = sequence.write(time, sample_ids)
					next_inputs = self.embedding.getEmbedding(sample_ids)
					return time + 1, next_inputs, cell_state, sequence

				def loop_prob(time, inputs, cell_state, seqence_logits):
					outputs, next_cell_state = self.cell(inputs, cell_state)
					logits = tf.add(tf.matmul(outputs, self.W), self.b)
					seqence_logits = seqence_logits.write(time, logits)
					next_inputs = self.ta_emb_reply.read(time)
					return time + 1, next_inputs, next_cell_state, seqence_logits

				time, inputs, cell_state, sequence, keepNumber = control_flow_ops.while_loop(
					cond=lambda time, _1, _2, _3, keepNumber: time < keepNumber,
					body=loop_keep,
					loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.cell.zero_state(self.batch_size, dtype=tf.float32), sequence, self.keepNumber))

				_, _, _, sequence = control_flow_ops.while_loop(
					cond=lambda time, _1, _2, _3: time < self.sequence_length,
					body=loop_gen,
					loop_vars=(time, inputs, cell_state, sequence))

				_, _, _, seqence_logits = control_flow_ops.while_loop(
					cond=lambda time, _1, _2, _3: time < self.sequence_length,
					body=loop_prob,
					loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.cell.zero_state(self.batch_size, dtype=tf.float32), seqence_logits))

				sequence = tf.transpose(sequence.stack(), perm=[1, 0])  # batch_size x seq_length
				seqence_logits = tf.transpose(seqence_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

		return sequence, seqence_logits


	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			self.buildInputGraph()

			self.seqence, self.seqence_logits = self.buildModel()

			self.buildTrainingGraph(self.seqence_logits)
			