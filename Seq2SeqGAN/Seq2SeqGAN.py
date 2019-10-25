import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Target import Target
from Discriminator import Discriminator
from Dataloader import GenDataLoader
from Dataloader import DiscDataLoader
from Dataloader import TargetDataLoader
from Dataloader import RealDataLoader
import numpy as np
import os
import time
import sys
import json
import random
import math
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)

experimentName = "real-data-target-and-pretrain-200-more-data-15-5-continued"
tensorboardDir = os.path.join(os.path.dirname(__file__), "tensorboard")
tensorboardDir = os.path.join(tensorboardDir, experimentName)
pathToStoreModelDir = os.path.join(os.path.dirname(__file__), "models")
pathToStoreModel = os.path.join(pathToStoreModelDir, "model.ckpt")
#logfile = "save/experiment-log-CNN.txt"
logfile = "save/"+experimentName+".txt"
SEED = 1337

class ConvergenceTracker():
	def __init__(self, slack):
		self.slack = slack
		self.reset()

	def reset(self):
		self.record = float("inf")
		self.timeSinceRecord = 0
		self.epochs = 0
		self.newEpoch()

	def newEpoch(self):
		self.loss = 0.0
		self.n = 0.0

	def report(self, loss):
		self.loss += loss
		self.n += 1.0

	def hasConverged(self):
		if self.n < 1.0:
			return False
		candidate = self.loss / self.n
		if candidate < self.record:
			self.record = candidate
			self.timeSinceRecord = 0
		self.timeSinceRecord += 1
		self.epochs += 1
		if self.timeSinceRecord >= self.slack:
			print(self.epochs)
			self.reset()
			return True
		self.newEpoch()
		return False

class GANChat():
	def __init__(self):
		pass

	def tensorboardWrite(self, writer, summary, iteration, writeToTensorboard):
		if writeToTensorboard:
			writer.add_summary(summary, iteration)

	def saveModel(self, sess, saver, epoch):
		if saver:
			if not os.path.exists(pathToStoreModelDir):
				os.makedirs(pathToStoreModelDir)
			#print("Saving model...")
			save_path = saver.save(sess, pathToStoreModel)
			print("Model saved ("+str(epoch)+"): " + save_path)

	def train(self):
		tf.reset_default_graph()
		random.seed(SEED)
		np.random.seed(SEED)
		tf.random.set_random_seed(SEED)
		self.batch_size = 64
		self.sequence_length = 64 # 20
		self.vocab_size = 30003 # 5000
		self.start_token = 0
		self.embedding_size = 32
		self.token_sample_rate = 16

		self.epochSize = 50000 #10000
		self.genPreTrainEpoch = 200 #120
		self.targetTrainingEpoch = 200 #120
		self.epochNumber = 200
		self.discPreTrainEpoch = 60 #50

		self.embeddingGEN = Embedding(self.vocab_size, self.embedding_size, "GEN")
		self.embeddingTAR = Embedding(self.vocab_size, self.embedding_size, "TAR")
		self.embeddingDISC = Embedding(self.vocab_size, self.embedding_size*2, "DISC")
		self.generator = Generator(self.embeddingGEN, self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.target = Target(self.embeddingTAR, self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.discriminator = Discriminator(self.embeddingDISC, self.sequence_length, self.start_token, self.batch_size)

		self.writeToTensorboard = True
		trainTarget = True
		trainTargetFakeData = False

		self.genDataLoader = GenDataLoader(self.batch_size, self.vocab_size, self.sequence_length)
		self.discDataLoader = DiscDataLoader(self.batch_size, self.vocab_size, self.sequence_length, self.genDataLoader)
		self.conTrack = ConvergenceTracker(15)
		saver = tf.train.Saver()

		dirname = os.path.dirname(logfile)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		log = open(logfile, 'w', buffering = 1)

		with tf.Session() as sess:
			if self.writeToTensorboard:
				writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
			else:
				writer = None
			print("Starting " + experimentName)
			print("Initialize new graph")
			sess.run(tf.global_variables_initializer())

			#print("load model")
			#saver.restore(sess, pathToStoreModel)
			#print("model loaded")
			#self.targetDataLoader = RealDataLoader(self.batch_size, self.epochSize)
			
			self.targetDataLoader = None
			if trainTarget:
				if trainTargetFakeData:
					self.targetDataLoader = self.trainTargetFakeData(sess, writer)
				else:
					self.targetDataLoader = self.trainTarget(sess, writer)
			
			print("Creating dataset")
			self.genDataLoader.createDataset(self.target, self.epochSize, sess, self.targetDataLoader)

			iteration = 0
			disc_iteration = 0
			
			print("PreTrain Generator")
			for epoch in range(self.genPreTrainEpoch):
				for _ in range(self.genDataLoader.num_batches):
					post, reply = self.genDataLoader.nextBatch()
					summary, genLoss = self.generator.pretrain(sess, post, reply)
					self.tensorboardWrite(writer, summary, iteration, self.writeToTensorboard)
					iteration+=1

				if epoch % 5 == 0:
					score = self.target.calculateScore(sess, self.generator, self.epochSize, self.targetDataLoader)
					print("PreTrain epoch {:>4}, score {:>6.3f} - ".format(epoch, score) + time.strftime("%H:%M:%S", time.localtime(time.time())))
					log.write(str(epoch) + " " + str(score) + '\n')
					self.saveModel(sess, saver, epoch)
			
			print("PreTrain Discriminator")
			epoch = 0
			while not self.conTrack.hasConverged():
				self.discDataLoader.createDataset(self.generator, self.epochSize, sess)
				for _ in range(5):
					for _ in range(self.discDataLoader.num_batches):
						post, reply, labels = self.discDataLoader.nextBatch()
						summary, discLoss = self.discriminator.train(sess, post, reply, labels)
						self.tensorboardWrite(writer, summary, disc_iteration, self.writeToTensorboard)
						disc_iteration+=1
						self.conTrack.report(discLoss)

				if epoch % 5 == 0:
					print("PreTrain epoch {:>4}, score {:>6.3f} - ".format(epoch, discLoss) + time.strftime("%H:%M:%S", time.localtime(time.time())))
					self.saveModel(sess, saver, epoch)
				epoch += 1

			iteration = 0
			print("Adverserial training")
			for epoch in range(self.genPreTrainEpoch, self.genPreTrainEpoch + self.epochNumber):

				#Generator
				for _ in range(1):
					post, reply = self.genDataLoader.getRandomBatch()
					fakeReply = self.generator.generate(sess, post)
					rewards = self.generator.calculateReward(sess, post, fakeReply, self.token_sample_rate, self.discriminator)
					summary, genLoss = self.generator.train(sess, post, fakeReply, rewards)
					self.tensorboardWrite(writer, summary, iteration, self.writeToTensorboard)
					iteration+=1

				if epoch % 5 == 0:
					score = self.target.calculateScore(sess, self.generator, self.epochSize, self.targetDataLoader)
					print("Ad Train epoch {:>4}, score {:>6.3f} - ".format(epoch, score) + time.strftime("%H:%M:%S", time.localtime(time.time())))
					log.write(str(epoch) + " " + str(score) + '\n')
					self.saveModel(sess, saver, epoch)
				
				#Discriminator
				for _ in range(15):
					self.discDataLoader.createDataset(self.generator, self.epochSize, sess)
					for _ in range(5):
						for _ in range(self.discDataLoader.num_batches):
							post, reply, labels = self.discDataLoader.nextBatch()
							summary, discLoss = self.discriminator.train(sess, post, reply, labels)
							self.tensorboardWrite(writer, summary, disc_iteration, self.writeToTensorboard)
							disc_iteration+=1
		
			log.close()

	def trainTarget(self, sess, writer):
		targetDataLoader = RealDataLoader(self.batch_size, self.epochSize)
		assert self.sequence_length == targetDataLoader.getSequenceLength()

		iteration = 0
		print("Training target")
		for epoch in range(self.targetTrainingEpoch):
			for _ in range(targetDataLoader.num_batches):
				post, reply = targetDataLoader.nextBatch()
				summary, loss = self.target.train(sess, post, reply)
				self.tensorboardWrite(writer, summary, iteration, self.writeToTensorboard)
				iteration+=1

			if epoch % 5 == 0:
				print("Target   epoch {:>4}, score {:>6.3f} - ".format(epoch, loss) + time.strftime("%H:%M:%S", time.localtime(time.time())))

		return targetDataLoader
					
	def trainTargetFakeData(self, sess, writer):
		targetDataLoader = TargetDataLoader()

		print("Creating Target training data")
		for _ in range(self.epochSize//self.batch_size):
			post = np.random.randint(self.vocab_size, size=(self.batch_size, self.sequence_length))
			reply = np.zeros((self.batch_size, self.sequence_length), dtype=int)
			for t in range(self.sequence_length):
				for b in range(self.batch_size):
					if t < 4:
						reply[b][t] = random.randint(0,self.vocab_size-1)
					else:
						tokens = random.choices(np.append(post[b][0:t], reply[b][t-4:t-1]), k=3)
						tokenId = random.choices(range(4), cum_weights=[0.3,0.6,0.9,1.0],k=1)[0]
						if tokenId == 3:
							reply[b][t] = random.randint(0,self.vocab_size-1)
						else:
							reply[b][t] = tokens[tokenId]
			targetDataLoader.appendBatch(post, reply)

		iteration = 0
		print("Training target")
		for epoch in range(self.targetTrainingEpoch):
			for _ in range(targetDataLoader.num_batches):
				post, reply = targetDataLoader.nextBatch()
				summary, loss = self.target.train(sess, post, reply)
				self.tensorboardWrite(writer, summary, iteration, self.writeToTensorboard)
				iteration+=1

			if epoch % 5 == 0:
				print("Target   epoch {:>4}, score {:>6.3f} - ".format(epoch, loss) + time.strftime("%H:%M:%S", time.localtime(time.time())))

		return targetDataLoader

				
if __name__ == "__main__":
	GANChat().train()