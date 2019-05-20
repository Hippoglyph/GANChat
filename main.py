import tensorflow as tf
from Embedding import Embedding
from Generator import Generator

class Trainer():
	def __init__(self):
		self.sequence_length = 4#64
		self.vocab_size = 5#30000 + 2
		self.embedding_size = 3#64

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, 0)

	def train(self):
		dummyInput = [[3,2,1,0],
						[0,1,2,3]]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			output = sess.run(
				[self.generator.encoder_final_state],
				{self.generator.input_x: dummyInput})

			print(output)

if __name__ == "__main__":
	Trainer().train()