import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq

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

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

class Discriminator():
	def __init__(self, embedding, sequence_length, start_token, learning_rate, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.encoder_units = 64
		self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
		self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
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
			self.embedded_reply_expanded = tf.expand_dims(self.embedded_reply, -1)

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

	def buildCNNModel(self):
		with tf.variable_scope("encoder"):
			std = 0.1
			features = []
			for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
				with tf.variable_scope("cov-"+str(filter_size)):
					filter_shape = [filter_size, self.embedding.getEmbeddingSize(), 1, num_filter]
					W = tf.Variable(tf.random_normal(filter_shape, stddev=std), name="W")
					b = tf.Variable(tf.random_normal([num_filter], stddev=std), name="b")
					conv = tf.nn.conv2d(self.embedded_reply_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
					pool = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], padding="VALID", name="pool", strides=[1,1,1,1])
					features.append(pool)
			total_features = sum(self.num_filters)
			combined_features = tf.concat(features, 3)
			combined_features_flat = tf.reshape(combined_features, [-1, total_features])

			features_highway = highway(combined_features_flat, combined_features_flat.get_shape()[1], 1, 0)
			dropout = tf.nn.dropout(features_highway, keep_prob=self.dropout_keep_prob)

		with tf.variable_scope("output"):
			std = 0.1
			W1 = tf.Variable(tf.random_normal([total_features, 2], stddev=std), name="W1")
			b1 = tf.Variable(tf.random_normal([2], stddev=std), name="b1")
			with tf.variable_scope("score"):
				score = tf.nn.relu(tf.add(tf.matmul(dropout, W1), b1))
			with tf.variable_scope("truth_prob"):
				truth_prob = tf.nn.softmax(score)[:,1]

		return score, truth_prob

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			
			self.buildInputGraph()

			#score, self.truth_prob = self.buildRNNModel()
			score, self.truth_prob = self.buildCNNModel()

			self.buildTrainingGraph(score)

			

