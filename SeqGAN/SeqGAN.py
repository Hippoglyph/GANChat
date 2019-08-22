import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Target import Target
from Discriminator import Discriminator
import numpy as np
import os
import time
import sys
import json
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)

tensorboardDir = os.path.join(os.path.dirname(__file__), "tensorboard")
tensorboardDir = os.path.join(tensorboardDir, "dummy")

class LossTracker():
	def __init__(self):
		self.startTime = time.time()
		self.printStamp = time.time()
		self.genLossNum = 0
		self.discLossNum = 0
		self.positiveBalanceNum = 0
		self.negativeBalanceNum = 0
		self.timePerItNum = 0
		self.genLossAcc = 0.0
		self.discLossAcc = 0.0
		self.positiveBalanceAcc = 0.0
		self.negativeBalanceAcc = 0.0
		self.timePerItAcc = 0.0
		self.printEvery = 60*10
		self.timeSinceLastLog = time.time()
		self.appendSeconds = 0
		self.suggestShutdown = False

	def printEverySecond(self, second):
		self.printEvery = second

	def addSeconds(self, seconds):
		self.appendSeconds = seconds

	def log(self, genLoss, discLoss, positiveBalance, negativeBalance, iteration):
		self.timePerItNum += 1
		self.timePerItAcc += time.time() - self.timeSinceLastLog
		self.timeSinceLastLog = time.time()
		if genLoss:
			self.genLossNum += 1
			self.genLossAcc += genLoss
		if discLoss:
			self.discLossNum += 1
			self.discLossAcc += discLoss
		if positiveBalance:
			self.positiveBalanceNum += 1
			self.positiveBalanceAcc += positiveBalance
		if negativeBalance:
			self.negativeBalanceNum += 1
			self.negativeBalanceAcc += negativeBalance
		if time.time() - self.printStamp >= self.printEvery:
			logString = "Iteration {:>7}".format(iteration)
			if genLoss:
				logString += ", GenLoss {:>5.3f}".format(self.genLossAcc/self.genLossNum)
			if discLoss:
				logString += ", DiscLoss {:>5.3f}".format(self.discLossAcc/self.discLossNum)
			if positiveBalance:
				logString += ", positiveBalance {:>5.3f}".format(self.positiveBalanceAcc/self.positiveBalanceNum)
			if negativeBalance:
				logString += ", negativeBalance {:>5.3f}".format(self.negativeBalanceAcc/self.negativeBalanceNum)
			logString += ", {:>6.3f} sec/iteration".format(self.timePerItAcc/self.timePerItNum)
			logString += ", hour {:>4}".format(int((time.time()-self.startTime + self.appendSeconds)/(60*60)))
			logString += ", Time " + time.strftime("%H:%M:%S", time.localtime(time.time()))
			print(logString)

			self.genLossNum = 0
			self.discLossNum = 0
			self.timePerItNum = 0
			self.genLossAcc = 0.0
			self.discLossAcc = 0.0
			self.timePerItAcc = 0.0
			self.printStamp = time.time()

class GANChat():
	def __init__(self):
		pass

	def tensorboardWrite(self, writer, summary, iteration, writeToTensorboard):
		if writeToTensorboard:
			writer.add_summary(summary, iteration)

	def train(self):
		tf.reset_default_graph()
		self.batch_size = 32
		self.sequence_length = 20
		self.vocab_size = 5000
		self.start_token = 0
		self.embedding_size = 32
		self.learning_rate = 0.0001
		self.token_sample_rate = 16

		self.epochSize = 10000
		self.genPreTrainEpoch = 120
		self.epochNumber = 200
		self.discPreTrainEpoch = 50

		self.embeddingGEN = Embedding(self.vocab_size, self.embedding_size, "GEN")
		self.embeddingTAR = Embedding(self.vocab_size, self.embedding_size, "TAR")
		self.embeddingDISC = Embedding(self.vocab_size, self.embedding_size*2, "DISC")
		self.generator = Generator(self.embeddingGEN, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.target = Target(self.embeddingTAR, self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.discriminator = Discriminator(self.embeddingDISC, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

		writeToTensorboard = True
		trainWithNoise = True

		lossTracker = LossTracker()
		genLoss = None
		discLoss = None
		positiveBalance = 0.5
		negativeBalance = 0.5
		iterationStart = 0
		currentIteration = 0
		storedModelTimestamp = time.time()
		with tf.Session() as sess:
			if writeToTensorboard:
				writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
			else:
				writer = None

			print("Initialize new graph")
			sess.run(tf.global_variables_initializer())

			iteration = 0
			for epoch in range(self.genPreTrainEpoch):
				for _ in range(self.epochSize//self.batch_size):
					batch = self.target.generate(sess)
					summary, genLoss = self.generator.pretrain(sess, batch)
					self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
					iteration+=1
					lossTracker.log(genLoss, discLoss, positiveBalance, negativeBalance, iteration)
			'''
			for epoch in range(self.discPreTrainEpoch):
				postBatch, replyBatch = self.data_loader.nextBatch()
				fakeSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
				realSequences = replyBatch

				negativeBalance = np.mean(self.discriminator.evaluate(sess, postBatch, fakeSequences))
				positiveBalance = np.mean(self.discriminator.evaluate(sess, postBatch, realSequences))

				posts =  np.concatenate([postBatch, postBatch])
				samples = np.concatenate([fakeSequences, realSequences])
				labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
				for _ in range(3):
					index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace = trainWithNoise)
					summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index])
				self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)

			for epoch in range(self.epochNumber):
				#Generator
				for _ in range(1):
					postBatch, replyBatch = self.data_loader.nextBatch()
					genSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
					rewards = self.generator.calculateReward(sess, postBatch, genSequences, self.token_sample_rate, self.discriminator)
					summary, genLoss = self.generator.train(sess, postBatch, genSequences, rewards)
				self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
				
				#Discriminator
				for _ in range(5):
					postBatch, replyBatch = self.data_loader.nextBatch()
					fakeSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
					realSequences = replyBatch

					negativeBalance = np.mean(self.discriminator.evaluate(sess, postBatch, fakeSequences))
					positiveBalance = np.mean(self.discriminator.evaluate(sess, postBatch, realSequences))

					posts =  np.concatenate([postBatch, postBatch])
					samples = np.concatenate([fakeSequences, realSequences])
					labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
					for _ in range(3):
						index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace=False)
						summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index])
					self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
			'''
				
if __name__ == "__main__":
	GANChat().train()