import tensorflow as tf

def LSTM_recurrent_unit(input_size,hidden_size, id, params):
	with tf.name_scope("LSTM"+str(id)):
		std = 0.1
		# Weights and Bias for input and hidden tensor
		Wi = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Wi")
		Ui = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Ui")
		bi = tf.Variable(tf.random_normal([hidden_size], stddev=std), name="LSTM_"+str(id)+"_Bi")

		Wf = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Wf")
		Uf = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Uf")
		bf = tf.Variable(tf.random_normal([hidden_size], stddev=std), name="LSTM_"+str(id)+"_Bf")

		Wog = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Wog")
		Uog = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Uog")
		bog = tf.Variable(tf.random_normal([hidden_size], stddev=std), name="LSTM_"+str(id)+"_Bog")

		Wc = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Wc")
		Uc = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=std), name="LSTM_"+str(id)+"_Uc")
		bc = tf.Variable(tf.random_normal([hidden_size], stddev=std), name="LSTM_"+str(id)+"_Bc")
		params.extend([
			Wi, Ui, bi,
			Wf, Uf, bf,
			Wog, Uog, bog,
			Wc, Uc, bc])

		def unit(x, hidden_memory_tm1):
			previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

			# Input Gate
			i = tf.sigmoid(
				tf.matmul(x, Wi) +
				tf.matmul(previous_hidden_state, Ui) + bi
			)

			# Forget Gate
			f = tf.sigmoid(
				tf.matmul(x, Wf) +
				tf.matmul(previous_hidden_state, Uf) + bf
			)

			# Output Gate
			o = tf.sigmoid(
				tf.matmul(x, Wog) +
				tf.matmul(previous_hidden_state, Uog) + bog
			)

			# New Memory Cell
			c_ = tf.nn.tanh(
				tf.matmul(x, Wc) +
				tf.matmul(previous_hidden_state, Uc) + bc
			)

			# Final Memory cell
			c = f * c_prev + i * c_

			# Current Hidden state
			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return unit

def LSTM_output_unit(input_size,hidden_size, id, params):
	with tf.name_scope("LSTM_output_"+str(id)):
		std = 0.1
		Wo = tf.Variable(tf.random_normal([hidden_size, input_size], stddev=std), name="LSTM_output_"+str(id)+"_Wo")
		bo = tf.Variable(tf.random_normal([input_size], stddev=std), name="LSTM_output_"+str(id)+"_Bo")
		params.extend([Wo, bo])

		def unit(hidden_memory_tuple):
			hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
			# hidden_state : batch x hidden_dim
			logits = tf.matmul(hidden_state, Wo) + bo
			# output = tf.nn.softmax(logits)
			return logits

		return unit