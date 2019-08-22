import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq

class Discriminator():
	def __init__(self, embedding, sequence_length, start_token, learning_rate, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 64
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dropout = 0.75
		self.scope_name = "discriminator"
		self.buildGraph()

	def train(self, sess, reply, labels):
		loss_summary, loss, _ = sess.run(
				[self.loss_summary, self.loss, self.update_params],
				{self.reply_seq: reply,self.targets: labels, self.dropout_keep_prob: self.dropout})
		return loss_summary, loss

	def evaluate(self, sess, reply):
		return sess.run(
			self.truth_prob,
			{self.reply_seq: reply})

	def buildInputGraph(self):
		with tf.variable_scope("inputs"):
			self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
			self.targets = tf.placeholder(tf.int32, shape=[self.batch_size], name="targets")
			self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
			#self.batch_size = tf.shape(self.post_seq)[0]
			#self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)


			self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)

	def buildTrainingGraph(self, score):
		with tf.variable_scope("train"):
			self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
			#self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
			#for r in self.discriminatorVariables:
				#print(r.name)
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=score))
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.discriminatorVariables), 5.0)
			self.update_params = optimizer.apply_gradients(zip(self.gradients, self.discriminatorVariables))
			self.loss_summary =  tf.summary.scalar("discriminator_loss", self.loss)


	def buildRNNModel(self):
		with tf.variable_scope("encoder"):

			reply_encoder_RNN = rnn.LSTMCell(self.encoder_units)

			_, reply_encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(reply_encoder_RNN, self.embedded_reply, dtype=tf.float32, initial_state=reply_encoder_RNN.zero_state(self.batch_size, dtype=tf.float32))

			encodedTensor = reply_encoder_final_hidden_memory_tuple[1]

		with tf.variable_scope("output"):
			std = 0.1
			W1 = tf.Variable(tf.random_normal([self.encoder_units, self.encoder_units//2], stddev=std), name="W1")
			b1 = tf.Variable(tf.random_normal([self.encoder_units//2], stddev=std), name="b1")
			W2 = tf.Variable(tf.random_normal([self.encoder_units//2, 2], stddev=std), name="W2")
			b2 = tf.Variable(tf.random_normal([2], stddev=std), name="b2")
			with tf.variable_scope("denseLayer1"):
				denseLayer1 = tf.nn.relu(tf.add(tf.matmul(encodedTensor, W1), b1)) # batch x encoder_units//2
			with tf.variable_scope("dropout"):
				denseLayer1Dropout = tf.nn.dropout(denseLayer1, keep_prob=self.dropout_keep_prob)
			with tf.variable_scope("score"):
				score = tf.add(tf.matmul(denseLayer1Dropout, W2), b2) # batch x 2
			with tf.variable_scope("truth_prob"):
				truth_prob = tf.nn.softmax(score)[:,1]

		return score, truth_prob

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			
			self.buildInputGraph()

			score, self.truth_prob = self.buildRNNModel()

			self.buildTrainingGraph(score)

			

