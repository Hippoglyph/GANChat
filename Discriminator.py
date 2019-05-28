import tensorflow as tf
from LSTM import LSTM_recurrent_unit, LSTM_output_unit
import numpy as np

class Discriminator():
	def __init__(self, embedding, sequence_length, start_token, learning_rate):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 4
		self.params = []
		self.learning_rate = learning_rate
		self.buildGraph()

	def train(self, sess, post, reply, labels):
		loss,_ = sess.run(
				[self.loss, self.update_params],
				{self.post_seq: post,self.reply_seq: reply,self.targets: labels})
		return loss

	def buildGraph(self):
		with tf.name_scope("discriminator"):
			with tf.name_scope("inputs"):
				self.post_seq = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="post_sequence")
				self.reply_seq = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="reply_sequence")
				self.targets = tf.placeholder(tf.int32, shape=[None], name="targets")

				self.batch_size = tf.shape(self.post_seq)[0]
				self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)

				self.params.extend(self.embedding.getParams())

				self.embedded_post = self.embedding.getEmbedding(self.post_seq)
				#with tf.device("/cpu:0"):
				self.embedded_post_transposed = tf.transpose(self.embedded_post, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

				self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)
				#with tf.device("/cpu:0"):
				self.embedded_reply_transposed = tf.transpose(self.embedded_reply, perm=[1, 0, 2]) # sequence_length x batch_size x embedding_size

			with tf.name_scope("encoder"):

				with tf.name_scope("post_encoder"):

					embedded_post_lt = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
					embedded_post_lt = embedded_post_lt.unstack(self.embedded_post_transposed)
					
					self.post_encoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units, "post_encoder_1", self.params)

					eh0 = tf.zeros([self.batch_size, self.encoder_units])
					eh0 = tf.stack([eh0, eh0])

					def post_encoder_loop(i, x_t, h_tm1):
						h_t = self.post_encoder_LSTM(x_t, h_tm1)
						return i + 1, embedded_post_lt.read(i), h_t

					_,_, self.post_encoder_final_hidden_memory_tuple = tf.while_loop(
						cond=lambda i, _1, _2: i < self.sequence_length,
						body=post_encoder_loop,
						loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), eh0))

				with tf.name_scope("reply_encoder"):

					embedded_reply_lt = tf.TensorArray(dtype=tf.float32, size=self.sequence_length)
					embedded_reply_lt = embedded_reply_lt.unstack(self.embedded_reply_transposed)
					
					self.reply_encoder_LSTM = LSTM_recurrent_unit(self.embedding.embedding_size,self.encoder_units, "reply_encoder_1", self.params)

					eh0 = tf.zeros([self.batch_size, self.encoder_units])
					eh0 = tf.stack([eh0, eh0])

					def reply_encoder_loop(i, x_t, h_tm1):
						h_t = self.reply_encoder_LSTM(x_t, h_tm1)
						return i + 1, embedded_reply_lt.read(i), h_t

					_,_, self.reply_encoder_final_hidden_memory_tuple = tf.while_loop(
						cond=lambda i, _1, _2: i < self.sequence_length,
						body=reply_encoder_loop,
						loop_vars= (tf.constant(0,dtype=tf.int32), self.embedding.getEmbedding(self.start_token), eh0))

				post_last_hidden = tf.unstack(self.post_encoder_final_hidden_memory_tuple)[0]
				reply_last_hidden = tf.unstack(self.reply_encoder_final_hidden_memory_tuple)[0]

				self.encodedTensor = tf.concat([post_last_hidden, reply_last_hidden], 1) # batch x (encoder_units + encoder_units)

				with tf.name_scope("output"):
					std = 0.1
					W1 = tf.Variable(tf.random_normal([self.encoder_units+self.encoder_units, self.encoder_units], stddev=std), name="W1")
					b1 = tf.Variable(tf.random_normal([self.encoder_units], stddev=std), name="b1")
					self.params.extend([W1,b1])
					W2 = tf.Variable(tf.random_normal([self.encoder_units, 2], stddev=std), name="W2")
					b2 = tf.Variable(tf.random_normal([2], stddev=std), name="b2")
					self.params.extend([W2,b2])
					with tf.name_scope("denseLayer1"):
						denseLayer1 = tf.add(tf.matmul(self.encodedTensor, W1), b1) # batch x encoder_units
					with tf.name_scope("score"):
						self.score = tf.add(tf.matmul(denseLayer1, W2), b2) # batch x 2
					with tf.name_scope("truth_prob"):
						self.truth_prob = tf.nn.softmax(self.score, 1)[:,1]

				with tf.name_scope("train"):
					self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.score))
					optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
					self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), 5.0)
					self.update_params = optimizer.apply_gradients(zip(self.gradients, self.params))