import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Discriminator import Discriminator
from TokenProcessor import TokenProcessor
from DataLoader import DataLoader
import numpy as np
import os
import time
import json
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

storeModelId = "LSTMpretrain"
loadModelId = "LSTMpretrain"
pathToModelsDir = os.path.join(os.path.dirname(__file__), "models")
pathToEvaluateDir = os.path.join(os.path.dirname(__file__), "evaluate")
pathToEvaluateDir = os.path.join(pathToEvaluateDir, storeModelId)
pathToStoreModelDir = os.path.join(pathToModelsDir, storeModelId)
pathToStoreModel = os.path.join(pathToStoreModelDir, "model.ckpt")
pathToLoadModelDir = os.path.join(pathToModelsDir, loadModelId)
pathToLoadModel = os.path.join(pathToLoadModelDir, "model.ckpt")

tensorboardDir = os.path.join(os.path.dirname(__file__), "tensorboard")
tensorboardDir = os.path.join(tensorboardDir, storeModelId)
iterationFile = "iteration"
timeFile = "trainingTime"

class MODE:
	preTrainGenerator = 0
	preTrainDiscriminator = 1
	adviserialTraining = 2

class LossTracker():
	def __init__(self):
		self.startTime = time.time()
		self.printStamp = time.time()
		self.genLossNum = 0
		self.discLossNum = 0
		self.timePerItNum = 0
		self.genLossAcc = 0.0
		self.discLossAcc = 0.0
		self.timePerItAcc = 0.0
		self.printEvery = 60
		self.timeSinceLastLog = time.time()
		self.appendSeconds = 0

	def addSeconds(self, seconds):
		self.appendSeconds = seconds

	def log(self, genLoss, discLoss, iteration, epoch):
		self.timePerItNum += 1
		self.timePerItAcc += time.time() - self.timeSinceLastLog
		self.timeSinceLastLog = time.time()
		if genLoss:
			self.genLossNum += 1
			self.genLossAcc += genLoss
		if discLoss:
			self.discLossNum += 1
			self.discLossAcc += discLoss
		if time.time() - self.printStamp >= self.printEvery:
			logString = "Iteration {:>7}".format(iteration)
			if genLoss:
				logString += ", GenLoss {:>5.3f}".format(self.genLossAcc/self.genLossNum)
			if discLoss:
				logString += ", DiscLoss {:>5.3f}".format(self.discLossAcc/self.discLossNum)
			logString += ", {:>6.3f} sec/iteration".format(self.timePerItAcc/self.timePerItNum)
			logString += ", epoch {:>4}".format(epoch)
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
		tf.reset_default_graph()
		self.batch_size = 8
		self.tokenProcessor = TokenProcessor()
		self.data_loader = DataLoader(self.batch_size)
		self.sequence_length = self.data_loader.getSequenceLength()
		self.vocab_size = self.tokenProcessor.getVocabSize()
		self.start_token = self.tokenProcessor.getStartToken()
		self.embedding_size = 32
		self.learning_rate = 0.0001
		self.token_sample_rate = 16
		self.storeModelEvery = 60*10
		self.timeStampLastSave = time.time()

	def saveModel(self, sess, saver, saveModel, iteration):
		if saveModel:
			if not os.path.exists(pathToStoreModelDir):
				os.makedirs(pathToStoreModelDir)
			print("Saving model...")
			save_path = saver.save(sess, pathToStoreModel)
			with open(os.path.join(pathToStoreModelDir, iterationFile), 'w', encoding="utf-8") as file:
				file.write(str(iteration))
			with open(os.path.join(pathToLoadModelDir, timeFile), 'r', encoding="utf-8") as file:
				totalSeconds = file.read()
			with open(os.path.join(pathToStoreModelDir, timeFile), 'w', encoding="utf-8") as file:
				file.write(str(int(int(time.time() - self.timeStampLastSave) + int(totalSeconds))))
			self.timeStampLastSave = time.time()
			print("Model saved ("+str(iteration)+"): " + save_path)

	def loadModel(self, sess, saver):
		if not os.path.exists(pathToLoadModelDir):
			print("loadModel: Model does not exit: " + pathToLoadModelDir)
			return
		print("Loading model...")
		saver.restore(sess, pathToLoadModel)
		with open(os.path.join(pathToLoadModelDir, iterationFile), 'r', encoding="utf-8") as file:
			iteration = file.read()
		with open(os.path.join(pathToLoadModelDir, timeFile), 'r', encoding="utf-8") as file:
			startime = file.read()
		print("Model loaded "+pathToLoadModel+ " (" + str(iteration)+")")
		return int(iteration), int(startime)

	def evaluate(self, sess, iteration, batch_size):
		print("Evaluating...")
		if not os.path.exists(pathToEvaluateDir):
			os.makedirs(pathToEvaluateDir)
		post, reply = self.data_loader.getTestBatch(batch_size)
		fakeReply = self.generator.generate(sess, post)

		dataPoints = []

		for i in range(batch_size):
			dataPoints.append({"post": self.tokenProcessor.sequenceToText(post[i]), "reply": self.tokenProcessor.sequenceToText(reply[i]), "fakereply": self.tokenProcessor.sequenceToText(fakeReply[i])})

		with open(os.path.join(pathToEvaluateDir, str(iteration)+".json"), 'w', encoding="utf-8") as file:
			for dataPoint in dataPoints:
				json.dump(dataPoint, file)
				file.write("\n")
		print("Evaluating stored (" + str(iteration)+")")

	def train(self):
		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

		trainingMode = MODE.preTrainGenerator
		loadModel = True
		saveModel = True
		evaluate = True

		saver = tf.train.Saver()

		lossTracker = LossTracker()
		genLoss = None
		discLoss = None
		iterationStart = 0
		currentIteration = 0
		storedModelTimestamp = time.time()
		with tf.Session() as sess:
			try:
				writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
				if loadModel:
					iterationStart, startTime = self.loadModel(sess, saver)
					lossTracker.addSeconds(startTime)
				else:
					print("Initialize new graph")
					sess.run(tf.global_variables_initializer())

				for iteration in range(iterationStart, 99999999):
					currentIteration = iteration

					if trainingMode == MODE.preTrainGenerator:
						postBatch, replyBatch = self.data_loader.nextBatch()
						summary, genLoss = self.generator.pretrain(sess, postBatch, replyBatch)
						writer.add_summary(summary, iteration)

					elif trainingMode == MODE.preTrainDiscriminator:
						postBatch, replyBatch = self.data_loader.nextBatch()
						fakeSequences = self.generator.generate(sess, postBatch)
						realSequences = replyBatch

						posts =  np.concatenate([postBatch, postBatch])
						samples = np.concatenate([fakeSequences, realSequences])
						labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
						for _ in range(3):
							index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace=False)
							summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index])
						writer.add_summary(summary, iteration)

					elif trainingMode == MODE.adviserialTraining:
						#Generator
						for _ in range(1):
							postBatch, replyBatch = self.data_loader.nextBatch()
							genSequences = self.generator.generate(sess, postBatch)
							rewards = self.generator.calculateReward(sess, postBatch, genSequences, self.token_sample_rate, self.discriminator)
							summary, genLoss = self.generator.train(sess, postBatch, genSequences, rewards)
						writer.add_summary(summary, iteration)
						
						#Discriminator
						for _ in range(5):
							postBatch, replyBatch = self.data_loader.nextBatch()
							fakeSequences = self.generator.generate(sess, postBatch)
							realSequences = replyBatch

							posts =  np.concatenate([postBatch, postBatch])
							samples = np.concatenate([fakeSequences, realSequences])
							labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
							for _ in range(3):
								index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace=False)
								summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index])
							writer.add_summary(summary, iteration)

					lossTracker.log(genLoss, discLoss, iteration, self.data_loader.getEpoch())

					if time.time() - storedModelTimestamp >= self.storeModelEvery: #Store every self.storeModelEvery second
						self.saveModel(sess, saver, saveModel, iteration)
						storedModelTimestamp = time.time()
						if evaluate:
							self.evaluate(sess, iteration, self.batch_size)

			except (KeyboardInterrupt, SystemExit):
				self.saveModel(sess, saver, saveModel, currentIteration)
				
				
if __name__ == "__main__":
	GANChat().train()