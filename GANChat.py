import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Discriminator import Discriminator
from TokenProcessor import TokenProcessor
from DataLoader import DataLoader
import numpy as np
import os
import time
import sys
import json
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)


#storeModelId = "ChoLSTMpretrain"
#loadModelId = "ChoLSTMpretrain"
#storeModelId = "BahdanauGRUpretrainN"
#loadModelId = "BahdanauGRUpretrainN"
storeModelId = "BahdanauGRU"
loadModelId = "BahdanauGRU"

#storeModelId = "BahdanauGRUDiscN"
#loadModelId = "BahdanauGRUDiscN"

pathToModelsDir = os.path.join(os.path.dirname(__file__), "models")
pathToEvaluateDir = os.path.join(os.path.dirname(__file__), "evaluate")
pathToEvaluateDummy = os.path.join(pathToEvaluateDir, "dummy")
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

	def log(self, genLoss, discLoss, positiveBalance, negativeBalance, iteration, epochProgress):
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
			logString += ", epoch {:>3.2f}%".format(epochProgress*100)
			logString += ", hour {:>4}".format(int((time.time()-self.startTime + self.appendSeconds)/(60*60)))
			logString += ", Time " + time.strftime("%H:%M:%S", time.localtime(time.time()))
			print(logString)

			if negativeBalance:
				self.suggestShutdown = self.positiveBalanceAcc/self.positiveBalanceNum - self.negativeBalanceAcc/self.negativeBalanceNum > 0.03

			self.genLossNum = 0
			self.discLossNum = 0
			self.timePerItNum = 0
			self.genLossAcc = 0.0
			self.discLossAcc = 0.0
			self.timePerItAcc = 0.0
			self.printStamp = time.time()
		return self.suggestShutdown

class GANChat():
	def __init__(self):
		pass

	def saveModel(self, sess, saver, saveModel, iteration):
		if saveModel:
			if not os.path.exists(pathToStoreModelDir):
				os.makedirs(pathToStoreModelDir)
				with open(os.path.join(pathToStoreModelDir, timeFile), 'w', encoding="utf-8") as file:
					file.write(str(0))
			print("Saving model...")
			save_path = saver.save(sess, pathToStoreModel)
			with open(os.path.join(pathToStoreModelDir, iterationFile), 'w', encoding="utf-8") as file:
				file.write(str(iteration))
			with open(os.path.join(pathToStoreModelDir, timeFile), 'r', encoding="utf-8") as file:
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

	def evaluate(self, sess, iteration, batch_size, noise = True, path = pathToEvaluateDir):
		if not os.path.exists(path):
			os.makedirs(path)
		post, reply = self.data_loader.getTestBatch(batch_size)
		fakeReply = self.generator.generate(sess, post, noise)

		dataPoints = []

		for i in range(batch_size):
			dataPoints.append({"post": self.tokenProcessor.sequenceToText(post[i]), "reply": self.tokenProcessor.sequenceToText(reply[i]), "fakereply": self.tokenProcessor.sequenceToText(fakeReply[i])})

		with open(os.path.join(path, str(iteration)+".json"), 'w', encoding="utf-8") as file:
			for dataPoint in dataPoints:
				json.dump(dataPoint, file)
				file.write("\n")
		print("Evaluating stored (" + str(iteration)+")")

	def tensorboardWrite(self, writer, summary, iteration, writeToTensorboard):
		if writeToTensorboard:
			writer.add_summary(summary, iteration)

	def infer(self):
		tf.reset_default_graph()
		self.batch_size = 32
		self.tokenProcessor = TokenProcessor()
		self.data_loader = DataLoader(self.batch_size, loadOnlyTest = True)
		self.sequence_length = self.data_loader.getSequenceLength()
		self.vocab_size = self.tokenProcessor.getVocabSize()
		self.start_token = self.tokenProcessor.getStartToken()
		self.embedding_size_Gen = 32
		self.embedding_size_Disc = 32
		self.learning_rate = 0.0001
		self.token_sample_rate = 8

		#self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding_size_Gen, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

		saver = tf.train.Saver()

		with tf.Session() as sess:
			iteration, _ = self.loadModel(sess, saver)
			self.evaluate(sess, iteration, self.batch_size, noise = True, path = pathToEvaluateDummy)


	def train(self):
		tf.reset_default_graph()
		self.batch_size = 32
		self.tokenProcessor = TokenProcessor()
		self.data_loader = DataLoader(self.batch_size)
		self.sequence_length = self.data_loader.getSequenceLength()
		self.vocab_size = self.tokenProcessor.getVocabSize()
		self.start_token = self.tokenProcessor.getStartToken()
		self.embedding_size = 32
		self.learning_rate = 0.0001
		self.token_sample_rate = 8
		self.storeModelEvery = 60*15
		self.timeStampLastSave = time.time()

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

		trainingMode = MODE.adviserialTraining
		loadModel = True
		saveModel = True
		evaluate = True
		writeToTensorboard = True
		autoBalance = True
		trainWithNoise = True
		shutdownWhenSuggested = False
		freezeDisc = True

		self.autoBalanceRange = 0.02

		saver = tf.train.Saver()

		lossTracker = LossTracker()
		genLoss = None
		discLoss = None
		positiveBalance = 0.5
		negativeBalance = 0.5
		iterationStart = 0
		currentIteration = 0
		storedModelTimestamp = time.time()
		with tf.Session() as sess:
			try:
				if writeToTensorboard:
					writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
				else:
					writer = None
				if loadModel:
					iterationStart, startTime = self.loadModel(sess, saver)
					lossTracker.addSeconds(startTime)
				else:
					print("Initialize new graph")
					sess.run(tf.global_variables_initializer())

				trainingString = "Adviserial Training"
				if trainingMode == MODE.preTrainGenerator:
					trainingString = "Pretraining Generator"
				elif trainingMode == MODE.preTrainDiscriminator:
					trainingString = "Pretraining Discriminator"
				print("Starting " + trainingString)

				for iteration in range(iterationStart, 99999999):
					currentIteration = iteration

					if trainingMode == MODE.preTrainGenerator:
						postBatch, replyBatch = self.data_loader.nextBatch()
						summary, genLoss = self.generator.pretrain(sess, postBatch, replyBatch, noise = trainWithNoise)
						self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)

					elif trainingMode == MODE.preTrainDiscriminator:
						postBatch, replyBatch = self.data_loader.nextBatch()
						fakeSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
						realSequences = replyBatch

						negativeBalance = np.mean(self.discriminator.evaluate(sess, postBatch, fakeSequences))
						positiveBalance = np.mean(self.discriminator.evaluate(sess, postBatch, realSequences))

						posts =  np.concatenate([postBatch, postBatch])
						samples = np.concatenate([fakeSequences, realSequences])
						labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
						for _ in range(3):
							index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace = False)
							summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index])
						self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)

					elif trainingMode == MODE.adviserialTraining:
						#Generator
						if not autoBalance or negativeBalance - positiveBalance < self.autoBalanceRange:
							for _ in range(1):
								postBatch, replyBatch = self.data_loader.nextBatch()
								genSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
								rewards = self.generator.calculateReward(sess, postBatch, genSequences, self.token_sample_rate, self.discriminator)
								summary, genLoss = self.generator.train(sess, postBatch, genSequences, rewards)
							self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)
						
						#Discriminator
						for _ in range(1):
							postBatch, replyBatch = self.data_loader.nextBatch()
							fakeSequences = self.generator.generate(sess, postBatch, noise = trainWithNoise)
							realSequences = replyBatch

							negativeBalance = np.mean(self.discriminator.evaluate(sess, postBatch, fakeSequences))
							positiveBalance = np.mean(self.discriminator.evaluate(sess, postBatch, realSequences))

							if not freezeDisc:

								gradientPenalty = 1.0 if not autoBalance or positiveBalance - negativeBalance < 0 else 1.0 - (positiveBalance - negativeBalance)/self.autoBalanceRange
								gradientPenalty = np.clip(gradientPenalty, 1e-20, 1.0)

								posts =  np.concatenate([postBatch, postBatch])
								samples = np.concatenate([fakeSequences, realSequences])
								labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
								for _ in range(3):
									index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace=False)
									summary, discLoss = self.discriminator.train(sess, posts[index], samples[index], labels[index], gradientPenalty)
								self.tensorboardWrite(writer, summary, iteration, writeToTensorboard)

					suggestShutDown = lossTracker.log(genLoss, discLoss, positiveBalance, negativeBalance, iteration, self.data_loader.getEpochProgress())

					if shutdownWhenSuggested and suggestShutDown:
						self.saveModel(sess, saver, saveModel, iteration)
						if evaluate:
							self.evaluate(sess, iteration, self.batch_size, noise = trainWithNoise)
						print("Imbalanced training - Forced shutdown")
						sys.exit(0)

					if time.time() - storedModelTimestamp >= self.storeModelEvery: #Store every self.storeModelEvery second
						self.saveModel(sess, saver, saveModel, iteration)
						storedModelTimestamp = time.time()
						if evaluate:
							self.evaluate(sess, iteration, self.batch_size, noise = trainWithNoise)

			except (KeyboardInterrupt, SystemExit):
				self.saveModel(sess, saver, saveModel, currentIteration)

	def play(self):
		tf.reset_default_graph()
		self.embedding = Embedding(6, 5)
		self.discriminator = Discriminator(self.embedding, 4, 0, 0.001, 3)
		self.generator = Generator(self.embedding, 4, 0, 6, 0.001, 3)

		dummyPost = [[0,1,2,3],
					[3,2,1,0],
					[2,2,3,1]]
		dummyReply = dummyPost

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#result = sess.run(
			#	[self.generator.seqences],
			#	{self.generator.post_seq: dummyPost, self.generator.reply_seq: dummyReply})
			result = self.discriminator.evaluate(sess, dummyPost, dummyReply)
			print(result)

			#result  = self.generator.calculateReward(sess, dummyPost, dummyReply, 5, self.discriminator)
			#result = self.discriminator.evaluate(sess, dummyPost, dummyReply)
			#print(result)

			#for _ in range(100):
				#_, loss = self.discriminator.train(sess, dummyPost, dummyReply, [0,1,0])
				#print(loss)

			#result = self.discriminator.evaluate(sess, dummyPost, dummyReply)
			#print(result)
			#result  = self.generator.calculateReward(sess, dummyPost, dummyReply, 5, self.discriminator)
			#print(result)
			#variables_names = [v.name for v in tf.trainable_variables()]
			#values = sess.run(variables_names)
			#for k,v in zip(variables_names, values):
			#	print(k)
				
if __name__ == "__main__":
	GANChat().train()
	#GANChat().infer()