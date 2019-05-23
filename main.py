import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)

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
		#dymmyTarget = [[3,2,1,0],
		#				[0,1,2,3]]
		config=tf.ConfigProto(log_device_placement=True)
		with tf.Session() as sess:
			print(device_lib.list_local_devices())
			sess.run(tf.global_variables_initializer())

			#output = self.generator.generate(sess, dummyInput)
			
			
			output = sess.run(
				[self.generator.pretrain_loss],
				{
				self.generator.input_seq: dummyInput,
				self.generator.target_seq: dummyInput
				})
			

			print(output)

if __name__ == "__main__":
	Trainer().train()