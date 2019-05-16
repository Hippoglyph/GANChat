import tensorflow as tf
from Embedding import Embedding
from Generator import Generator

class Trainer():
	def __init__(self):
		self.sequence_length = 3#64
		self.vocab_size = 4#30000 + 2
		self.embedding_size = 2#64

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length)

	def train(self):
		dummyInput = [[0,1,2]]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			output = sess.run(
				[self.generator.embedded_input],
				{self.generator.input_x: dummyInput})

			print(output)

if __name__ == "__main__":
	Trainer().train()