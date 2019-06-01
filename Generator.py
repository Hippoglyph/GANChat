import tensorflow as tf
from LSTM import LSTM_recurrent_unit, LSTM_output_unit
import numpy as np

class Generator():
	def __init__(self, embedding, sequence_length, start_token, vocab_size, learning_rate, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 4
		self.noiseSize = 2
		self.noiseStd = 1.0
		self.decoder_units = self.encoder_units + self.noiseSize
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.params = []
		self.learning_rate = learning_rate
		self.buildGraph()

	def generate(self, sess, post, noise=True):
		if(noise):
			output = sess.run(
					self.seqences[0],
					{self.post_seq: post,self.reply_seq: post})
		else:
			noise = np.zeros((self.batch_size, self.noiseSize))
			output = sess.run(
					self.seqences[0],
					{self.post_seq: post,self.reply_seq: post, self.noise: noise})
		return output

	def pretrain(self, sess, post, reply):
		noise = np.zeros((self.batch_size, self.noiseSize))
		loss_summary,_ = sess.run(
				[self.pretrain_summary, self.pretrain_update],
				{self.post_seq: post,self.reply_seq: reply, self.noise: noise})
		return loss_summary

	def rolloutStep(self, sess, post, reply, keepIndex):
		return sess.run(
			self.seqences[keepIndex],
			{self.post_seq: post, self.reply_seq: reply})

	def calculateReward(self, sess, post, reply, tokenSampleRate, discriminator):
		rewards = np.zeros((self.batch_size, self.sequence_length))
		for keepNumber in range(self.sequence_length):
			for i in range(tokenSampleRate):
				sampleReply = self.rolloutStep(sess, post, reply, keepNumber)
				rewards[:,keepNumber] += discriminator.evaluate(sess, post, sampleReply)
		return rewards / tokenSampleRate

	def train(self, sess, post, reply, rewards):
		loss_summary,_ = sess.run(
				[self.train_summary, self.update],
				{self.post_seq: post, self.reply_seq: reply, self.rewards: rewards})
		return loss_summary

	def buildGraph(self):
		with tf.name_scope("generator"):
			with tf.name_scope("inputs"):
				self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
				self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
				self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length], name="rewards")
				self.noise = tf.placeholder_with_default(tf.random_normal([self.batch_size, self.noiseSize], 0.0, self.noiseStd), shape=[self.batch_size, self.noiseSize], name="noise")
			
				#self.batch_size = tf.shape(self.post_seq)[0]
				self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
			
				self.params.extend(self.embedding.getParams())

				self.embedded_post = self.embedding.getEmbedding(self.post_seq)
				#with tf.device("/cpu:0"):
				self.embedded_post_transposed = tf.transpose(self.embedded_post, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

			with tf.name_scope("encoder"):

				embedded_post_lt = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
				embedded_post_lt = embedded_post_lt.unstack(self.embedded_post_transposed)
				
				self.encoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units, "encoder_1", self.params)

				eh0 = tf.zeros([self.batch_size, self.encoder_units])
				eh0 = tf.stack([eh0, eh0])

				def encoder_loop(i, x_t, h_tm1):
					h_t = self.encoder_LSTM(x_t, h_tm1)
					return i + 1, embedded_post_lt.read(i), h_t

				_,_, self.encoder_final_hidden_memory_tuple = tf.while_loop(
					cond=lambda i, _1, _2: i < self.sequence_length,
					body=encoder_loop,
					loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), eh0))

			with tf.name_scope("decoder"):

				self.decoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.decoder_units, "decoder_1",self.params)
				self.decoder_LSTM_output = LSTM_output_unit(self.vocab_size,self.decoder_units, "decoder_1", self.params)

				with tf.name_scope("noise"):
					h0, c0 = tf.unstack(self.encoder_final_hidden_memory_tuple)
					h0 = tf.concat([h0, self.noise], 1)
					c0 = tf.concat([c0, self.noise], 1)
					decoderH0 = tf.stack([h0,c0])

				with tf.name_scope("generate"):

					reply_lt = tf.TensorArray(dtype=tf.int32, size=self.sequence_length)
					reply_lt = reply_lt.unstack(tf.transpose(self.reply_seq, perm=[1,0]))

					seqences = []
					seqences_logits = []
					for iteration_number in range(self.sequence_length+1):
						gen_seq = tf.TensorArray(dtype=tf.int32, size=self.sequence_length)
						gen_seq_logits = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)

						def gen_loop(i, x_t, h_tm1, gen_seq, gen_seq_logits, keep_length):
							h_t = self.decoder_LSTM(x_t, h_tm1)
							o_t = self.decoder_LSTM_output(h_t) # batch_size x vocab_size
							prob = tf.nn.softmax(o_t)
							next_token = tf.cond(i >= keep_length, lambda: tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self.batch_size]), lambda: reply_lt.read(i))
							x_tp1 = self.embedding.getEmbedding(next_token)
							gen_seq = gen_seq.write(i, next_token)
							gen_seq_logits = gen_seq_logits.write(i, o_t)
							return i + 1, x_tp1, h_t, gen_seq, gen_seq_logits, keep_length

						_,_,_,gen_seq, gen_seq_logits,_ = tf.while_loop(
							cond=lambda i, _1, _2, _3, _4, _5: i < self.sequence_length,
							body=gen_loop,
							loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), decoderH0, gen_seq, gen_seq_logits, tf.constant(iteration_number,dtype=tf.int32)))

						seqences.append(tf.transpose(gen_seq.stack(), perm=[1,0])) # batch_size x sequence_length
						seqences_logits.append(tf.transpose(gen_seq_logits.stack(), perm=[1,0,2])) # batch_size x sequence_length x vocab_size

					self.seqences = seqences
					self.seqences_logits = seqences_logits

			with tf.name_scope("pretrain"):
				#logits = tf.log(tf.clip_by_value(self.seqences_logits[self.sequence_length], 1e-8,1-1e-8))
				self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reply_seq, logits=self.seqences_logits[self.sequence_length]))
				pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.params), 5.0)
				self.pretrain_update = pretrain_optimizer.apply_gradients(zip(self.pretrain_grad, self.params))
				self.pretrain_summary = tf.summary.scalar("generator_pretrain_loss", self.pretrain_loss)

			with tf.name_scope("train"):
				genSequence = self.seqences[self.sequence_length]
				genProb = tf.nn.softmax(self.seqences_logits[self.sequence_length])
				genLogProb = tf.log(tf.clip_by_value(genProb, 1e-8,1-1e-8))
				self.loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(genSequence, self.vocab_size) * genLogProb, -1) * self.rewards)
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), 5.0)
				self.update = optimizer.apply_gradients(zip(self.grad, self.params))
				self.train_summary = tf.summary.scalar("generator_loss", self.loss)

			


	
	
	
				