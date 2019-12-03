import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from Embedding import Embedding

'''
class DecoderHelper(Helper):
	def __init__(self, keep_length, sequence_length, embedding, reply_seq_embedding, start_token_embedding, _batch_size):
		self.keep_length = keep_length
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
		prob = tf.nn.softmax(outputs)
		return tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self._batch_size])

	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished= tf.greater_equal(time+1, self.sequence_length)
		if self.keep_length <= 0:
			next_token = self.embedding.getEmbedding(sample_ids)
		else:
			next_token = tf.cond(tf.logical_or(tf.greater_equal(time+1, self.keep_length), finished), lambda: self.embedding.getEmbedding(sample_ids), lambda: self.reply_embedding[:,time,:])
		return finished, next_token, state
'''
class Generator():
	def __init__(self, embedding_size, sequence_length, start_token_symbol, vocab_size, batch_size):
		self.sequence_length = sequence_length
		self.start_token_symbol = start_token_symbol
		self.encoder_units = 1024
		self.noiseSize = 128
		self.noiseStd = 1.0
		self.loss_size = 1000.0
		self.decoder_units = self.encoder_units + self.noiseSize
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = 1e-2
		self.embedding_size = embedding_size
		self.scope_name = "generator"
		self.buildGraph()

	def generate(self, sess, post, noise=True):
		if(noise):
			output = sess.run(
					self.sequence,
					{self.post_seq: post})
		else:
			noise = np.zeros((self.batch_size, self.noiseSize))
			output = sess.run(
					self.sequence,
					{self.post_seq: post, self.noise: noise})
		return output

	def pretrain(self, sess, post, reply, noise=True):
		if(noise):
			loss_summary, loss, _ = sess.run(
					[self.pretrain_summary, self.pretrain_loss, self.pretrain_update],
					{self.post_seq: post, self.reply_seq: reply})
		else:
			noise = np.zeros((self.batch_size, self.noiseSize))
			loss_summary, loss, _ = sess.run(
					[self.pretrain_summary, self.pretrain_loss, self.pretrain_update],
					{self.post_seq: post, self.reply_seq: reply, self.noise: noise})
		return loss_summary, loss

	def rolloutStep(self, sess, post, reply, keepIndex):
		return sess.run(
			self.sequence,
			{self.post_seq: post, self.reply_seq: reply, self.keepNumber: keepIndex})

	def calculateReward(self, sess, post, reply, tokenSampleRate, discriminator):
		rewards = np.zeros((self.batch_size, self.sequence_length))
		for keepNumber in range(1,self.sequence_length):
			for _ in range(tokenSampleRate):
				sample_reply = self.rolloutStep(sess, post, reply, keepNumber)
				rewards[:,keepNumber-1] += discriminator.evaluate(sess, post, sample_reply)
		rewards[:,-1] = discriminator.evaluate(sess, post, reply) * tokenSampleRate
		return rewards / tokenSampleRate

	def train(self, sess, post, reply, rewards):
		loss_summary, loss, _ = sess.run(
				[self.train_summary, self.loss, self.update],
				{self.post_seq: post, self.reply_seq: reply, self.rewards: rewards})
		return loss_summary, loss

	def buildInputGraph(self):
		self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
		self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
		self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length], name="rewards")
		self.noise = tf.placeholder_with_default(tf.random_normal([self.batch_size, self.noiseSize], 0.0, self.noiseStd), shape=[self.batch_size, self.noiseSize], name="noise")
		self.keepNumber = tf.placeholder_with_default(0, shape=())
		self.embedding = Embedding(self.vocab_size, self.embedding_size, "Generator")
	
		self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token_symbol,dtype=tf.int32)
		self.embedded_start_token = self.embedding.getEmbedding(self.start_token)

		self.embedded_post = self.embedding.getEmbedding(self.post_seq)
		self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)

		self.ta_reply_seq = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
		self.ta_reply_seq = self.ta_reply_seq.unstack(tf.transpose(self.reply_seq, perm=[1, 0])) #seq_length x batch

		self.ta_emb_post_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
		self.ta_emb_post_seq = self.ta_emb_post_seq.unstack(tf.transpose(self.embedded_post, perm=[1, 0, 2])) #seq_length x batch x embedding

		self.ta_emb_reply_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
		self.ta_emb_reply_seq = self.ta_emb_reply_seq.unstack(tf.transpose(self.embedded_reply, perm=[1, 0, 2])) #seq_length x batch x embedding

	def buildTrainingGraph(self, logits):
		self.generatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
		#for r in self.generatorVariables:
			#print(r.name)
			#print(r.shape)

		#Pretrain
		self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reply_seq, logits=logits))
		pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.generatorVariables), 5.0)
		self.pretrain_update = pretrain_optimizer.apply_gradients(zip(self.pretrain_grad, self.generatorVariables))
		self.pretrain_summary = tf.summary.scalar("generator_pretrain_loss", self.pretrain_loss)

		#RL-Learning
		genProb = tf.nn.softmax(logits)
		genLogProb = tf.log(tf.clip_by_value(genProb, 1e-20, 1.0))
		self.loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(self.reply_seq, self.vocab_size) * genLogProb, -1) * self.rewards)*self.loss_size
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.generatorVariables), 5.0)
		self.update = optimizer.apply_gradients(zip(self.grad, self.generatorVariables))
		self.train_summary = tf.summary.scalar("generator_loss", self.loss)

	def buildBahdanauModel(self):
		'''
		self.encoder_RNN_fw = rnn.GRUCell(self.encoder_units//2)
		self.encoder_RNN_bw = rnn.GRUCell(self.encoder_units//2)

		(encoder_outputs_fw, encoder_outputs_bw), encoder_final_state = tf.nn.bidirectional_dynamic_rnn(encoder_RNN_fw, encoder_RNN_bw, self.embedded_post, dtype=tf.float32, initial_state_fw=encoder_RNN_fw.zero_state(self.batch_size, dtype=tf.float32),initial_state_bw=encoder_RNN_bw.zero_state(self.batch_size, dtype=tf.float32))

		encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], 2)


		decoder_RNN = rnn.GRUCell(self.decoder_units)
		decoder_RNN_projection = tf.layers.Dense(units=self.vocab_size, use_bias=True, kernel_initializer=tf.initializers.random_normal(0,0.1))

		attention = seq2seq.BahdanauAttention(num_units=self.decoder_units, memory=encoder_outputs)

		attention_RNN = seq2seq.AttentionWrapper(decoder_RNN, attention, attention_layer_size=self.decoder_units//2)


		decoderH0 = tf.zeros((self.batch_size, self.encoder_units), dtype=tf.float32)
		decoderH0 = tf.concat([decoderH0, self.noise], 1) # batch_size x decoder_units


		seqences = []
		seqences_logits = []
		for iteration_number in range(self.sequence_length+1):
			decoder = seq2seq.BasicDecoder(
				cell=attention_RNN,
				helper=DecoderHelper(iteration_number, self.sequence_length, self.embedding, self.embedded_reply, self.embedded_start_token, self.batch_size),
				initial_state=attention_RNN.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=decoderH0),
				output_layer=decoder_RNN_projection
				)

			final_outputs, _, _ = seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=self.sequence_length)

			seqences.append(tf.reshape(final_outputs.sample_id, [self.batch_size, self.sequence_length])) # batch_size x sequence_length
			seqences_logits.append(tf.reshape(final_outputs.rnn_output,[self.batch_size, self.sequence_length, self.vocab_size])) # batch_size x sequence_length x vocab_size
		'''

		######
		std = 0.1
		self.encoder_RNN_fw = rnn.GRUCell(self.encoder_units//2, kernel_initializer=tf.initializers.random_normal(0, std), name="encoder_RNN_fw")
		self.encoder_RNN_bw = rnn.GRUCell(self.encoder_units//2, kernel_initializer=tf.initializers.random_normal(0, std), name="encoder_RNN_bw")
		self.decoder_RNN = rnn.GRUCell(self.decoder_units, kernel_initializer=tf.initializers.random_normal(0, std), name="decoder_RNN")
		self.decoder_W = tf.Variable(tf.random_normal([self.decoder_units//2, self.vocab_size], stddev=std), name="decoder_W")
		self.decoder_b = tf.Variable(tf.random_normal([self.vocab_size], stddev=std), name="decoder_b")

		sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
		sequence_logits = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

		ta_encoder_outputs_fw = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)#, clear_after_read=False)
		ta_encoder_outputs_bw = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)#, clear_after_read=False)

		def loop_encoder_fw(time, inputs, cell_state, ta_encoder_outputs_fw):
			_, next_cell_state = self.encoder_RNN_fw(inputs, cell_state)
			ta_encoder_outputs_fw = ta_encoder_outputs_fw.write(time, next_cell_state)
			next_inputs = self.ta_emb_post_seq.read(time)
			return time + 1, next_inputs, next_cell_state, ta_encoder_outputs_fw

		def loop_encoder_bw(time, inputs, cell_state, ta_encoder_outputs_bw):
			_, next_cell_state = self.encoder_RNN_bw(inputs, cell_state)
			ta_encoder_outputs_bw = ta_encoder_outputs_bw.write(time, next_cell_state)
			next_inputs = self.ta_emb_post_seq.read(time)
			return time - 1, next_inputs, next_cell_state, ta_encoder_outputs_bw

		_, _,_, ta_encoder_outputs_fw = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time < self.sequence_length,
			body=loop_encoder_fw,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.encoder_RNN_fw.zero_state(self.batch_size, dtype=tf.float32), ta_encoder_outputs_fw))

		_, _,_, ta_encoder_outputs_bw = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time >= 0,
			body=loop_encoder_bw,
			loop_vars=(tf.constant(self.sequence_length - 1, dtype=tf.int32), self.embedded_start_token, self.encoder_RNN_bw.zero_state(self.batch_size, dtype=tf.float32), ta_encoder_outputs_bw))

		encoder_outputs_fw = tf.transpose(ta_encoder_outputs_fw.stack(), perm=[1, 0, 2])  # batch_size x seq_length x units
		encoder_outputs_bw = tf.transpose(ta_encoder_outputs_bw.stack(), perm=[1, 0, 2])  # batch_size x seq_length x units
		print(encoder_outputs_fw)
		print(encoder_outputs_bw)
		encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], 2)
		print(encoder_outputs)
		#encoder_outputs = encoder_outputs_fw
		attention = seq2seq.BahdanauAttention(num_units=self.decoder_units, memory=encoder_outputs)
		self.attention_RNN = seq2seq.AttentionWrapper(self.decoder_RNN, attention, attention_layer_size=self.decoder_units//2)

		decoderH0 = tf.zeros((self.batch_size, self.encoder_units), dtype=tf.float32)
		decoderH0 = tf.concat([decoderH0, self.noise], 1) # batch_size x decoder_units

		def loop_keep(time, inputs, cell_state, sequence, keepNumber):
			_, next_cell_state = self.attention_RNN(inputs, cell_state)
			next_inputs = self.ta_emb_reply_seq.read(time)
			sequence = sequence.write(time, self.ta_reply_seq.read(time))
			return time + 1, next_inputs, next_cell_state, sequence, keepNumber

		def loop_gen(time, inputs, cell_state, sequence):
			outputs, next_cell_state = self.attention_RNN(inputs, cell_state)
			logits = tf.add(tf.matmul(outputs, self.decoder_W), self.decoder_b)
			prob = tf.nn.softmax(logits)
			sample_ids = tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self.batch_size])
			sequence = sequence.write(time, sample_ids)
			next_inputs = self.embedding.getEmbedding(sample_ids)
			return time + 1, next_inputs, next_cell_state, sequence

		def loop_prob(time, inputs, cell_state, sequence_logits):
			outputs, next_cell_state = self.attention_RNN(inputs, cell_state)
			logits = tf.add(tf.matmul(outputs, self.decoder_W), self.decoder_b)
			sequence_logits = sequence_logits.write(time, logits)
			next_inputs = self.ta_emb_reply_seq.read(time)
			return time + 1, next_inputs, next_cell_state, sequence_logits

		time, inputs, cell_state, sequence, keepNumber = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3, keepNumber: time < keepNumber,
			body=loop_keep,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.attention_RNN.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=decoderH0), sequence, self.keepNumber))

		_, _, _, sequence = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time < self.sequence_length,
			body=loop_gen,
			loop_vars=(time, inputs, cell_state, sequence))

		_, _, _, sequence_logits = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time < self.sequence_length,
			body=loop_prob,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.attention_RNN.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=decoderH0), sequence_logits))

		sequence = tf.transpose(sequence.stack(), perm=[1, 0])  # batch_size x seq_length
		sequence_logits = tf.transpose(sequence_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

		######

		return sequence, sequence_logits

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			self.buildInputGraph()

			self.sequence, self.sequence_logits = self.buildBahdanauModel()

			self.buildTrainingGraph(self.sequence_logits)
			