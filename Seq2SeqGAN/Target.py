import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn, seq2seq
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Target():
	def __init__(self, embedding, sequence_length, start_token, vocab_size, batch_size):
		self.sequence_length = sequence_length
		self.start_token = start_token
		self.embedding = embedding
		self.units = 32
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate_MLE = 1e-2
		self.scope_name = "target"
		self.buildGraph()

	def generate(self, sess, post):
		output = sess.run(
				self.sequence,
				{self.post_seq: post})
		return output

	def getProbability(self, sess, post, reply):
		output = sess.run(
				self.score,
				{self.post_seq: post, self.reply_seq: reply})
		return output

	def calculateScore(self, sess, generator, total_iteration):
		nll = []
		for _ in range(total_iteration//self.batch_size):
			post = np.random.randint(self.vocab_size, size=(self.batch_size, self.sequence_length))
			sequence = generator.generate(sess, post)
			nll.append(self.getProbability(sess, post, sequence))
		return np.mean(nll)

	def train(self, sess, post, reply, rewards):
		loss_summary, loss, _ = sess.run(
				[self.train_summary, self.loss, self.update],
				{self.post_seq: post, self.reply_seq: reply, self.rewards: rewards})
		return loss_summary, loss

	def buildInputGraph(self):
		self.post_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="post_sequence")
		self.reply_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name="reply_sequence")
		self.start_token = tf.cast(tf.ones([self.batch_size])*self.start_token,dtype=tf.int32)
		self.embedded_start_token = self.embedding.getEmbedding(self.start_token)
		self.embedded_post_seq = self.embedding.getEmbedding(self.post_seq)
		self.embedded_reply_seq = self.embedding.getEmbedding(self.reply_seq)

		self.ta_emb_post_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
		self.ta_emb_post_seq = self.ta_emb_post_seq.unstack(tf.transpose(self.embedded_post_seq, perm=[1, 0, 2])) #seq_length x batch x embedding

		self.ta_emb_reply_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
		self.ta_emb_reply_seq = self.ta_emb_reply_seq.unstack(tf.transpose(self.embedded_reply_seq, perm=[1, 0, 2])) #seq_length x batch x embedding

	def buildTrainingGraph(self, logits):
		self.generatorVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.embedding.getNameScope())
		#for r in self.generatorVariables:
			#print(r.name)
			#print(r.shape)

		#Pretrain
		self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reply_seq, logits=logits))
		pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_MLE)
		self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.generatorVariables), 5.0)
		self.pretrain_update = pretrain_optimizer.apply_gradients(zip(self.pretrain_grad, self.generatorVariables))
		self.pretrain_summary = tf.summary.scalar("target_loss", self.pretrain_loss)

	def buildModel(self):
		std = 1.0
		self.encoderCell = rnn.LSTMCell(self.units, initializer=tf.initializers.random_normal(0, std), name="LSTMEncoder")
		self.decoderCell = rnn.LSTMCell(self.units, initializer=tf.initializers.random_normal(0, std), name="LSTMDecoder")
		self.W = tf.Variable(tf.random_normal([self.units, self.vocab_size], stddev=std), name="W")
		self.b = tf.Variable(tf.random_normal([self.vocab_size], stddev=std), name="b")

		sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
		sequence_logits = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

		def loop_encoder(time, inputs, cell_state):
			_, next_cell_state = self.encoderCell(inputs, cell_state)
			next_inputs = self.ta_emb_post_seq.read(time)
			return time + 1, next_inputs, next_cell_state

		def loop_gen(time, inputs, cell_state, sequence):
			outputs, next_cell_state = self.decoderCell(inputs, cell_state)
			logits = tf.add(tf.matmul(outputs, self.W), self.b)
			prob = tf.nn.softmax(logits)
			sample_ids = tf.reshape(tf.multinomial(tf.log(prob), 1, output_dtype=tf.int32), [self.batch_size])
			sequence = sequence.write(time, sample_ids)
			next_inputs = self.embedding.getEmbedding(sample_ids)
			return time + 1, next_inputs, next_cell_state, sequence

		def loop_prob(time, inputs, cell_state, sequence_logits):
			outputs, next_cell_state = self.decoderCell(inputs, cell_state)
			logits = tf.add(tf.matmul(outputs, self.W), self.b)
			sequence_logits = sequence_logits.write(time, logits)
			next_inputs = self.ta_emb_reply_seq.read(time)
			return time + 1, next_inputs, next_cell_state, sequence_logits

		_, _, encoder_cell_state = control_flow_ops.while_loop(
			cond=lambda time, _1, _2: time < self.sequence_length,
			body=loop_encoder,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, self.encoderCell.zero_state(self.batch_size, dtype=tf.float32)))

		_, _, _, sequence = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time < self.sequence_length,
			body=loop_gen,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, encoder_cell_state, sequence))

		_, _, _, sequence_logits = control_flow_ops.while_loop(
			cond=lambda time, _1, _2, _3: time < self.sequence_length,
			body=loop_prob,
			loop_vars=(tf.constant(0, dtype=tf.int32), self.embedded_start_token, encoder_cell_state, sequence_logits))

		sequence = tf.transpose(sequence.stack(), perm=[1, 0])  # batch_size x seq_length
		sequence_logits = tf.transpose(sequence_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

		return sequence, sequence_logits

	def buildGraph(self):
		with tf.variable_scope(self.scope_name):
			self.buildInputGraph()

			self.sequence, self.sequence_logits = self.buildModel()

			self.buildTrainingGraph(self.sequence_logits)

			self.probs = tf.nn.softmax(self.sequence_logits)

			self.score = -tf.reduce_sum(tf.one_hot(self.reply_seq, self.vocab_size) * tf.log(self.probs))/(self.sequence_length * self.batch_size)