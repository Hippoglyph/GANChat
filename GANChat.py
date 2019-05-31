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
		self.batch_size = 2
		self.dropout = 0.75

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

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

			for _ in range(1000):

			
				output = self.generator.pretrain(sess, dummyInput, dummyInput)
			#output = self.discriminator.train(sess, dummyInput, dummyInput, dummyLabels, self.dropout)
			#output = self.discriminator.train(sess, dummyInput, dummyInput, dummyLabels, self.dropout)
				print(output)
			#output = self.discriminator.evaluate(sess, dummyInput, dummyInput)
			output = self.generator.generate(sess, dummyInput, noise=False)
			print(output)
			#output = self.generator.rolloutStep(sess, dummyInput, dummyInput, 1)
			#output = self.generator.calculateReward(sess, dummyInput, dummyInput, 5, self.discriminator)
			#rewards = self.generator.calculateReward(sess, dummyInput, dummyInput, 5, self.discriminator)
			#output = self.generator.train(sess, dummyInput, dummyInput, rewards)
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
				

				
				
if __name__ == "__main__":
	GANChat().train()