import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from Embedding import Embedding
import math

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
	'''
	Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
	Args:
	input_: a tensor or a list of 2D, batch x n, Tensors.
	output_size: int, second dimension of W[i].
	scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
	A 2D Tensor with shape [batch x output_size] equal to
	sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
	ValueError: if some of the arguments has unspecified or wrong shape.
  '''

	shape = input_.get_shape().as_list()
	if len(shape) != 2:
		raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	if not shape[1]:
		raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	input_size = shape[1]

	# Now the computation.
	with tf.variable_scope(scope or "SimpleLinear"):
		matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
		bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

	return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
	"""Highway Network (cf. http://arxiv.org/abs/1505.00387).
	t = sigmoid(Wy + b)
	z = t * g(Wy + b) + (1 - t) * y
	where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
	"""
	size = input_.get_shape()[1]
	with tf.variable_scope(scope):
		for idx in range(num_layers):
			g = f(linear(input_, size, scope='highway_lin_%d' % idx))

			t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

			output = t * g + (1. - t) * input_
			input_ = output

	return output

def context_gating(input_layer, scope=None):
		"""Context Gating (https://arxiv.org/pdf/1706.06905.pdf)
		Args:
		input_layer: Input layer in the following shape:
		'batch_size' x 'number_of_activation'
		Returns:
		activation: gated layer in the following shape:
		'batch_size' x 'number_of_activation'
		"""

		input_dim = input_layer.get_shape().as_list()[1] 
		
		with tf.variable_scope(scope or "Context_Gating"):
			gating_weights = tf.get_variable("gating_weights",
			  [input_dim, input_dim],
			  initializer = tf.random_normal_initializer(
			  stddev = 1.0 / math.sqrt(input_dim)))
			
	 
			gating_biases = tf.get_variable("gating_biases",
			[input_dim],
			initializer = tf.random_normal_initializer(stddev = 1.0 / math.sqrt(input_dim)))

		gates = tf.matmul(input_layer, gating_weights)
		gates += gating_biases

		gates = tf.sigmoid(gates)

		activation = tf.multiply(input_layer,gates)

		return activation

def feed_forward(input_layer, output_dim, activation=None, scope=None):

	input_dim = input_layer.get_shape().as_list()[1]
	with tf.variable_scope(scope or "feed forward"):
		weights = tf.get_variable("weights",
			  [input_dim, output_dim],
			  initializer = tf.random_normal_initializer(
			  stddev = 1.0 / math.sqrt(input_dim)))
		biases = tf.get_variable("biases",
			[output_dim],
			initializer = tf.random_normal_initializer(stddev = 1.0 / math.sqrt(output_dim)))

	unactivated = tf.add(tf.matmul(input_layer, weights), biases)

	if activation is not None:
		return activation(unactivated)

	return unactivated

class Discriminator():
	def __init__(self, sequence_length, vocab_size, batch_size):
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embedding_size = 64
		self.encoder_units = 512
		self.batch_size = batch_size
		self.learning_rate = 1e-4
		self.dropout = 0.75
		self.grad_clip = 5.0
		self.l2_lambda = 0.2
		self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
		self.num_filters = [200, 300, 300, 300, 300, 200, 200, 200, 200, 200, 260, 260]
		self.scope_name = "discriminator"
		self.buildGraph()

	def train(self, sess, post, reply, labels):
		loss_summary, loss, _ = sess.run(
				[self.loss_summary, self.loss, self.update_params],
				{self.post_seq: post, self.reply_seq: reply, self.targets: labels, self.dropout_keep_prob: self.dropout})
		return loss_summary, loss

	def evaluate(self, sess, post, reply):
		return sess.run(
			self.truth_prob,
			{self.post_seq: post, self.reply_seq: reply})

	def buildInputGraph(self):
		self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
		self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
		self.targets = tf.placeholder(tf.int32, shape=[self.batch_size], name="targets")
		self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
		self.embedding = Embedding(self.vocab_size, self.embedding_size, "Discriminator")

		self.embedded_reply = self.embedding.getEmbedding(self.reply_seq)
		self.embedded_reply_expanded = tf.expand_dims(self.embedded_reply, -1)
		self.embedded_post = self.embedding.getEmbedding(self.post_seq)
		self.embedded_post_expanded = tf.expand_dims(self.embedded_post, -1)
		self.l2_loss = tf.constant(0.0)

	def buildTrainingGraph(self, score):
		self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
		#self.discriminatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
		#for r in self.discriminatorVariables:
			#print(r.name)
			#print(r.shape)

		entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=score)
		self.loss = tf.reduce_mean(entropy) + self.l2_lambda * self.l2_loss
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.discriminatorVariables), self.grad_clip)
		self.update_params = optimizer.apply_gradients(zip(self.gradients, self.discriminatorVariables))
		self.loss_summary =  tf.summary.scalar("discriminator_loss", self.loss)

	def buildRNNModel(self):

		#POST Encoder
		post_encoder_RNN = rnn.LSTMCell(self.encoder_units, name="Post_RNN")
		_, post_encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(post_encoder_RNN, self.embedded_post, dtype=tf.float32, initial_state=post_encoder_RNN.zero_state(self.batch_size, dtype=tf.float32))

		#REPLY Encoder
		reply_encoder_RNN = rnn.LSTMCell(self.encoder_units, name="Reply_RNN")
		_, reply_encoder_final_hidden_memory_tuple = tf.nn.dynamic_rnn(reply_encoder_RNN, self.embedded_reply, dtype=tf.float32, initial_state=reply_encoder_RNN.zero_state(self.batch_size, dtype=tf.float32))
		
		post_last_hidden = post_encoder_final_hidden_memory_tuple[1]
		reply_last_hidden = reply_encoder_final_hidden_memory_tuple[1]

		encodedTensor = tf.concat([post_last_hidden, reply_last_hidden], 1) # batch x (encoder_units + encoder_units)

		denseLayer1 = feed_forward(encodedTensor, self.encoder_units, activation=tf.nn.relu, scope="denseLayer1")
		denseLayer1Dropout = tf.nn.dropout(denseLayer1, keep_prob=self.dropout_keep_prob)
		score = feed_forward(denseLayer1Dropout, 2, activation=None, scope="score")
		truth_prob = tf.nn.softmax(score)[:,1]

		return score, truth_prob

	def buildCNNModel(self):
		std = 0.1

		#post
		features_post = []
		for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
			with tf.variable_scope("cov-"+str(filter_size)+"-post"):
				filter_shape = [filter_size, self.embedding.getEmbeddingSize(), 1, num_filter]
				W = tf.Variable(tf.random_normal(filter_shape, stddev=std), name="W")
				b = tf.Variable(tf.random_normal([num_filter], stddev=std), name="b")
				conv = tf.nn.conv2d(self.embedded_post_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pool = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], padding="VALID", name="pool", strides=[1,1,1,1])
				features_post.append(pool)
		total_features_post = sum(self.num_filters)
		combined_features_post = tf.concat(features_post, 3)
		combined_features_post_flat = tf.reshape(combined_features_post, [-1, total_features_post])

		#reply
		features_reply = []
		for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
			with tf.variable_scope("cov-"+str(filter_size)+"-reply"):
				filter_shape = [filter_size, self.embedding.getEmbeddingSize(), 1, num_filter]
				W = tf.Variable(tf.random_normal(filter_shape, stddev=std), name="W")
				b = tf.Variable(tf.random_normal([num_filter], stddev=std), name="b")
				conv = tf.nn.conv2d(self.embedded_reply_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pool = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], padding="VALID", name="pool", strides=[1,1,1,1])
				features_reply.append(pool)
		total_features_reply = sum(self.num_filters)
		combined_features_reply = tf.concat(features_reply, 3)
		combined_features_reply_flat = tf.reshape(combined_features_reply, [-1, total_features_reply])

		features = tf.concat([combined_features_post_flat, combined_features_reply_flat], 1)

		cg1 = context_gating(features, scope="Context-Gateing_1")
		featuresScaled = feed_forward(cg1, (total_features_post + total_features_reply)//2, activation=None, scope="F1")
		features_highway = highway(featuresScaled, 2, 0)
		cg2 = context_gating(features_highway, scope="Context-Gateing_2")
		dropout = tf.nn.dropout(cg2, keep_prob=self.dropout_keep_prob)

		score = feed_forward(dropout, 2, activation=None, scope="score")
		truth_prob = tf.nn.softmax(score)[:,1]

		return score, truth_prob

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			
			self.buildInputGraph()

			#score, self.truth_prob = self.buildRNNModel()
			score, self.truth_prob = self.buildCNNModel()

			self.buildTrainingGraph(score)