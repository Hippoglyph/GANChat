import tensorflow as tf
from Embedding import Embedding
from Generator import Generator

class Trainer():
	def __init__(self):
		self.sequence_length = 4#64
		self.vocab_size = 5#30000 + 2
		self.embedding_size = 3#64
		self.start_token = 0

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size)

	def train(self):
		dummyInput = [[3,2,1,0],
						[0,1,2,3]]
		dymmyTarget = [[3,2,1,0],
						[0,1,2,3]]
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			#output = self.generator.generate(sess, dummyInput)
			
			
			output = sess.run(
				[self.generator.pretrain_loss],
				{
				self.generator.input_seq: dummyInput,
				self.generator.target_seq: dymmyTarget
				})
			

			print(output)

if __name__ == "__main__":
	Trainer().train()