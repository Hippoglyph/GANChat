import tensorflow as tf
from LSTM import LSTM_recurrent_unit, LSTM_output_unit
import numpy as np
from tensorflow.contrib import rnn, seq2seq

class Discriminator():
	def __init__(self, embedding, sequence_length, start_token, learning_rate, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 512
		self.batch_size = batch_size
		self.params = []
		self.learning_rate = learning_rate
		self.dropout = 0.75
		self.scope_name = "discriminator"
		self.buildGraph()

	def train(self, sess, post, reply, labels):
		loss_summary, loss, _ = sess.run(
				[self.loss_summary, self.loss, self.update_params],
				{self.post_seq: post,self.reply_seq: reply,self.targets: labels, self.dropout_keep_prob: self.dropout})
		return loss_summary, loss

	def evaluate(self, sess, post, reply):
		return sess.run(
			self.truth_prob,
			{self.post_seq: post, self.reply_seq: reply})

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			with tf.variable_scope("inputs"):
				self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
				self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
				self.targets = tf.placeholder(tf.int32, shape=[self.batch_size], name="targets")
				self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
				#self.batch_size = tf.shape(self.post_seq)[0]
				self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)

				self.params.extend(self.embedding.getParams())

				self.embedded_post = self.embedding.getEmbedding(self.post_seq)

				self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)


			with tf.variable_scope("encoder"):

				with tf.variable_scope("post_encoder"):

					self.post_encoder_RNN = rnn.LSTMCell(self.encoder_units)

					_, self.post_encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(self.post_encoder_RNN, self.embedded_post, dtype=tf.float32, initial_state=self.post_encoder_RNN.zero_state(self.batch_size, dtype=tf.float32))

				with tf.variable_scope("reply_encoder"):

					self.reply_encoder_RNN = rnn.LSTMCell(self.encoder_units)

					_, self.reply_encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(self.reply_encoder_RNN, self.embedded_reply, dtype=tf.float32, initial_state=self.reply_encoder_RNN.zero_state(self.batch_size, dtype=tf.float32))
				
				post_last_hidden = self.post_encoder_final_hidden_memory_tuple[1]
				reply_last_hidden = self.reply_encoder_final_hidden_memory_tuple[1]

				self.encodedTensor = tf.concat([post_last_hidden, reply_last_hidden], 1) # batch x (encoder_units + encoder_units)

			with tf.variable_scope("output"):
				std = 0.1
				W1 = tf.Variable(tf.random_normal([self.encoder_units+self.encoder_units, self.encoder_units], stddev=std), name="W1")
				b1 = tf.Variable(tf.random_normal([self.encoder_units], stddev=std), name="b1")
				W2 = tf.Variable(tf.random_normal([self.encoder_units, 2], stddev=std), name="W2")
				b2 = tf.Variable(tf.random_normal([2], stddev=std), name="b2")
				with tf.variable_scope("denseLayer1"):
					denseLayer1 = tf.nn.relu(tf.add(tf.matmul(self.encodedTensor, W1), b1)) # batch x encoder_units
				with tf.variable_scope("dropout"):
					denseLayer1Dropout = tf.nn.dropout(denseLayer1, keep_prob=self.dropout_keep_prob)
				with tf.variable_scope("score"):
					self.score = tf.add(tf.matmul(denseLayer1Dropout, W2), b2) # batch x 2
				with tf.variable_scope("truth_prob"):
					self.truth_prob = tf.nn.softmax(self.score, 1)[:,1]

			with tf.variable_scope("train"):
				#self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
				self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
				#for r in self.discriminatorVariables:
					#print(r.name)
				self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.score))
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.discriminatorVariables), 5.0)
				self.update_params = optimizer.apply_gradients(zip(self.gradients, self.discriminatorVariables))
				self.loss_summary =  tf.summary.scalar("discriminator_loss", self.loss)