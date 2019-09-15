import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Target import Target
from Discriminator import Discriminator
from Dataloader import GenDataLoader
from Dataloader import DiscDataLoader
import numpy as np
import os
import time
import sys
import json
import random
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)

tensorboardDir = os.path.join(os.path.dirname(__file__), "tensorboard")
tensorboardDir = os.path.join(tensorboardDir, "CNN-mean")
#logfile = "save/experiment-log-CNN.txt"
logfile = "save/experiment-log-CNN-mean.txt"
SEED = 1337


class GANChat():
	def __init__(self):
		pass

	def tensorboardWrite(self, writer, summary, iteration, writeToTensorboard):
		if writeToTensorboard:
			writer.add_summary(summary, iteration)

	def train(self):
		tf.reset_default_graph()
		random.seed(SEED)
		np.random.seed(SEED)
		tf.random.set_random_seed(SEED)
		self.batch_size = 64
		self.sequence_length = 20
		self.vocab_size = 5000
		self.start_token = 0
		self.embedding_size = 32
		self.token_sample_rate = 16

		self.epochSize = 10000
		self.genPreTrainEpoch = 120
		self.epochNumber = 200
		self.discPreTrainEpoch = 50

		self.embeddingGEN = Embedding(self.vocab_size, self.embedding_size, "GEN")
		self.embeddingTAR = Embedding(self.vocab_size, self.embedding_size, "TAR")
		self.embeddingDISC = Embedding(self.vocab_size, self.embedding_size*2, "DISC")
		self.generator = Generator(self.embeddingGEN, self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.target = Target(self.embeddingTAR, self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.discriminator = Discriminator(self.embeddingDISC, self.sequence_length, self.start_token, self.batch_size)

		writeToTensorboard = True

		self.genDataLoader = GenDataLoader(self.batch_size)
		self.discDataLoader = DiscDataLoader(self.batch_size, self.genDataLoader)

		dirname = os.path.dirname(logfile)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		log = open(logfile, 'w')

		with tf.Session() as sess:
			if writeToTensorboard:
				writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
			else:
				writer = None

			print("Initialize new graph")
			sess.run(tf.global_variables_initializer())

			print("Creating dataset")
			self.genDataLoader.createDataset(self.target, self.epochSize, sess)

			iteration = 0
			print("PreTrain Generator")
			for epoch in range(self.genPreTrainEpoch):
				for _ in range(self.genDataLoader.num_batches):
					batch = self.genDataLoader.nextBatch()
					summary, genLoss = self.generator.pretrain(sess, batch)
					self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
					iteration+=1

				if epoch % 5 == 0:
					score = self.target.calculateScore(sess, self.generator, self.epochSize)
					print("PreTrain epoch {:>4}, score {:>6.3f} - ".format(epoch, score) + time.strftime("%H:%M:%S", time.localtime(time.time())))
					log.write(str(epoch) + " " + str(score) + '\n')
			
			disc_iteration = 0
			print("PreTrain Discriminator")
			for epoch in range(self.discPreTrainEpoch):
				self.discDataLoader.createDataset(self.generator, self.epochSize, sess)
				for _ in range(3):
					for _ in range(self.discDataLoader.num_batches):
						batch, labels = self.discDataLoader.nextBatch()
						summary, discLoss = self.discriminator.train(sess, batch, labels)
						self.tensorboardWrite(writer, summary, disc_iteration, writeToTensorboard)
						disc_iteration+=1

				if epoch % 5 == 0:
					print("PreTrain epoch {:>4}, score {:>6.3f} - ".format(epoch, discLoss) + time.strftime("%H:%M:%S", time.localtime(time.time())))

			disc_iteration += 1000
			iteration = 0
			print("Adverserial training")
			for epoch in range(self.genPreTrainEpoch, self.genPreTrainEpoch + self.epochNumber):
				
				#Generator
				for _ in range(1):
					genSequences = self.generator.generate(sess)
					rewards = self.generator.calculateReward(sess, genSequences, self.token_sample_rate, self.discriminator)
					summary, genLoss = self.generator.train(sess, genSequences, rewards)
					self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
					iteration+=1
				
				#Discriminator
				for _ in range(5):
					self.discDataLoader.createDataset(self.generator, self.epochSize, sess)
					for _ in range(3):
						for _ in range(self.discDataLoader.num_batches):
							batch, labels = self.discDataLoader.nextBatch()
							summary, discLoss = self.discriminator.train(sess, batch, labels)
							self.tensorboardWrite(writer, summary, disc_iteration, writeToTensorboard)
							disc_iteration+=1

				if epoch % 5 == 0:
					score = self.target.calculateScore(sess, self.generator, self.epochSize)
					print("Ad Train epoch {:>4}, score {:>6.3f} - ".format(epoch, score) + time.strftime("%H:%M:%S", time.localtime(time.time())))
					log.write(str(epoch) + " " + str(score) + '\n')

			log.close()
				
if __name__ == "__main__":
	GANChat().train()