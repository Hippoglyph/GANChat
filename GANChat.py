import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Discriminator import Discriminator
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class GANChat():
	def __init__(self):
		self.sequence_length = 4#64
		self.vocab_size = 5#30000 + 2
		self.embedding_size = 3#64
		self.start_token = 0
		self.learning_rate = 0.01

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate)

	def train(self):
		dummyLabels = [0, 1]
		dummyInput = [[3,2,1,0],
						[0,1,2,3]]
		#dymmyTarget = [[3,2,1,0],
		#				[0,1,2,3]]
		#config=tf.ConfigProto(log_device_placement=True)
		with tf.Session() as sess:
			#print(device_lib.list_local_devices())
			sess.run(tf.global_variables_initializer())

			for _ in range(100):

				#output = self.generator.generate(sess, dummyInput)
				#output = self.generator.pretrain(sess, dummyInput, dummyInput)
				output = self.discriminator.train(sess, dummyInput, dummyInput, dummyLabels)
				#output = self.generator.rolloutStep(sess, dummyInput, dummyInput, 1)
				'''
				output = sess.run(
					self.discriminator.testOutput,
					{
					self.discriminator.post_seq: dummyInput,
					self.discriminator.reply_seq: dummyInput,
					self.discriminator.targets: dummyLabels
					})
				'''
				'''
				output = sess.run(
					[self.generator.pretrain_loss],
					{
					self.generator.post_seq: dummyInput,
					self.generator.target_seq: dummyInput
					})
				'''
				print(output)

				
				
if __name__ == "__main__":
	GANChat().train()