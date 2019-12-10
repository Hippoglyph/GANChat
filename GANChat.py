import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
from TokenProcessor import TokenProcessor
from DataLoader import DataLoader
from DataLoader import DiscDataLoader
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
storeModelId = "BahdanauGRUNew"
loadModelId = "BahdanauGRUNew"

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
IterFile = "Iteration"
TBPreTrainGenIterFile = "TBPreTrainGenIter"
TBGenIterFile = "TBGenIter"
TBDiscIterFile = "TBDiscIter"
timeFile = "trainingTime"

class MODE:
	preTrainGenerator = 0
	preTrainDiscriminator = 1
	adversarialTraining = 2

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
		self.printEvery = 15#60*60*1
		self.timeSinceLastLog = time.time()
		self.appendSeconds = 0

	def printEverySecond(self, second):
		self.printEvery = second

	def addSeconds(self, seconds):
		self.appendSeconds = seconds

	def log(self, genLoss, discLoss, iteration, epochProgress):
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
			logString += ", epoch {:>3.2f}%".format(epochProgress*100)
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
		return

class GANChat():
	def __init__(self):
		pass

	def saveModel(self, sess, saver, saveModel, iteration, TBPreTrainGenIter, TBGenIter, TBDiscIter):
		if saveModel:
			if not os.path.exists(pathToStoreModelDir):
				os.makedirs(pathToStoreModelDir)
				with open(os.path.join(pathToStoreModelDir, timeFile), 'w', encoding="utf-8") as file:
					file.write(str(0))
			print("Saving model...")
			save_path = saver.save(sess, pathToStoreModel)
			with open(os.path.join(pathToStoreModelDir, IterFile), 'w', encoding="utf-8") as file:
				file.write(str(iteration))
			with open(os.path.join(pathToStoreModelDir, TBPreTrainGenIterFile), 'w', encoding="utf-8") as file:
				file.write(str(TBPreTrainGenIter))
			with open(os.path.join(pathToStoreModelDir, TBGenIterFile), 'w', encoding="utf-8") as file:
				file.write(str(TBGenIter))
			with open(os.path.join(pathToStoreModelDir, TBDiscIterFile), 'w', encoding="utf-8") as file:
				file.write(str(TBDiscIter))
			with open(os.path.join(pathToStoreModelDir, timeFile), 'r', encoding="utf-8") as file:
				totalSeconds = file.read()
			with open(os.path.join(pathToStoreModelDir, timeFile), 'w', encoding="utf-8") as file:
				file.write(str(int(int(time.time() - self.timeStampLastSave) + int(totalSeconds))))
			self.timeStampLastSave = time.time()
			print("Model saved (" +time.strftime("%H:%M:%S", time.localtime(time.time()))+")" + save_path)

	def loadModel(self, sess, saver):
		if not os.path.exists(pathToLoadModelDir):
			print("loadModel: Model does not exit: " + pathToLoadModelDir)
			return
		print("Loading model...")
		saver.restore(sess, pathToLoadModel)
		with open(os.path.join(pathToLoadModelDir, IterFile), 'r', encoding="utf-8") as file:
			iteration = file.read()
		with open(os.path.join(pathToLoadModelDir, TBPreTrainGenIterFile), 'r', encoding="utf-8") as file:
			TBPreTrainGenIter = file.read()
		with open(os.path.join(pathToLoadModelDir, TBGenIterFile), 'r', encoding="utf-8") as file:
			TBGenIter = file.read()
		with open(os.path.join(pathToLoadModelDir, TBDiscIterFile), 'r', encoding="utf-8") as file:
			TBDiscIter = file.read()
		with open(os.path.join(pathToLoadModelDir, timeFile), 'r', encoding="utf-8") as file:
			startime = file.read()
		print("Model loaded "+pathToLoadModel+ " (" + str(iteration)+")")
		return int(iteration), int(TBPreTrainGenIter), int(TBGenIter), int(TBDiscIter), int(startime)

	def evaluate(self, sess, iteration, batch_size, path = pathToEvaluateDir):
		if not os.path.exists(path):
			os.makedirs(path)
		post, reply = self.data_loader.getTestBatch(batch_size)
		fakeReply = self.generator.generate(sess, post)

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

	'''
	def infer(self):
		tf.reset_default_graph()
		self.batch_size = 32
		self.tokenProcessor = TokenProcessor()
		self.data_loader = DataLoader(self.batch_size, loadOnlyTest = True)
		self.sequence_length = self.data_loader.getSequenceLength()
		self.vocab_size = self.tokenProcessor.getVocabSize()
		self.start_token = self.tokenProcessor.getStartToken()
		self.embedding_size_Disc = 64
		self.learning_rate = 0.0001
		self.token_sample_rate = 16

		self.embedding = Embedding(self.vocab_size, self.embedding_size_Disc)
		self.generator = Generator(self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

		saver = tf.train.Saver()

		with tf.Session() as sess:
			iteration, _ = self.loadModel(sess, saver)
			self.evaluate(sess, iteration, self.batch_size, noise = True, path = pathToEvaluateDummy)
	'''


	def train(self):
		tf.reset_default_graph()
		self.batch_size = 64
		self.tokenProcessor = TokenProcessor()
		self.data_loader = DataLoader(self.batch_size)
		self.sequence_length = self.data_loader.getSequenceLength()
		self.vocab_size = self.tokenProcessor.getVocabSize()
		self.start_token = self.tokenProcessor.getStartToken()
		self.token_sample_rate = 16
		self.storeModelEvery = 60*60*1
		self.discEpochSize = 50000
		self.timeStampLastSave = time.time()

		self.generator = Generator(self.sequence_length, self.start_token, self.vocab_size, self.batch_size)
		self.discriminator = Discriminator(self.sequence_length, self.vocab_size, self.batch_size)
		self.disc_data_loader = DiscDataLoader(self.data_loader, self.generator, self.discEpochSize)

		trainingMode = MODE.preTrainGenerator
		loadModel = False
		saveModel = False
		evaluate = False
		writeToTensorboard = False

		saver = tf.train.Saver()

		lossTracker = LossTracker()
		genLoss = None
		discLoss = None

		iterationStart = 0
		currentIteration = 0
		TBPreTrainGenIter = 0
		TBGenIter = 0
		TBDiscIter = 0
		storedModelTimestamp = time.time()

		discIteration = -1
		adversarialIteration = 0
		with tf.Session() as sess:
			try:
				if writeToTensorboard:
					writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
				else:
					writer = None
				if loadModel:
					iterationStart, TBPreTrainGenIter, TBGenIter, TBDiscIter, startTime = self.loadModel(sess, saver)
					lossTracker.addSeconds(startTime)
				else:
					print("Initialize new graph")
					sess.run(tf.global_variables_initializer())

				trainingString = "Adversarial Training"
				if trainingMode == MODE.preTrainGenerator:
					trainingString = "Pretraining Generator"
				elif trainingMode == MODE.preTrainDiscriminator:
					trainingString = "Pretraining Discriminator"
				print("Starting " + trainingString)

				for iteration in range(iterationStart, 99999999):
					currentIteration = iteration

					if trainingMode == MODE.preTrainGenerator:
						post, reply = self.data_loader.nextBatch()
						summary, genLoss = self.generator.pretrain(sess, post, reply)
						self.tensorboardWrite(writer, summary, TBPreTrainGenIter, writeToTensorboard)
						TBPreTrainGenIter += 1

					elif trainingMode == MODE.preTrainDiscriminator:
						discIteration = (discIteration+1) % (5*(self.discEpochSize//self.batch_size)) #Train 5 times on same disc epoch
						if discIteration == 0:
							self.disc_data_loader.createDataset(sess)
						post, reply, labels = self.disc_data_loader.nextBatch()
						summary, discLoss = self.discriminator.train(sess, post, reply, labels)
						self.tensorboardWrite(writer, summary, TBDiscIter, writeToTensorboard)
						TBDiscIter += 1

					elif trainingMode == MODE.adversarialTraining:

						if adversarialIteration == 0: #Train generator only once every 15 disc epoch
							#Generator
							post, reply = self.data_loader.nextBatch()
							fakeReply = self.generator.generate(sess, post)
							rewards = self.generator.calculateReward(sess, post, fakeReply, self.token_sample_rate, self.discriminator)
							summary, genLoss = self.generator.train(sess, post, fakeReply, rewards)
							self.tensorboardWrite(writer, summary, TBGenIter, writeToTensorboard)
							TBGenIter += 1
						
						#Discriminator
						if adversarialIteration % ((self.discEpochSize//self.batch_size)*5) == 0: #Train 5 times on same disc epoch
							self.disc_data_loader.createDataset(sess)
						post, reply, labels = self.disc_data_loader.nextBatch()
						summary, discLoss = self.discriminator.train(sess, post, reply, labels)
						self.tensorboardWrite(writer, summary, TBDiscIter, writeToTensorboard)
						TBDiscIter += 1

						adversarialIteration = (adversarialIteration+1) % ((self.discEpochSize//self.batch_size)*5*15)

					lossTracker.log(genLoss, discLoss, iteration, self.data_loader.getEpochProgress())

					if time.time() - storedModelTimestamp >= self.storeModelEvery: #Store every self.storeModelEvery second
						self.saveModel(sess, saver, saveModel, iteration, TBPreTrainGenIter, TBGenIter, TBDiscIter)
						storedModelTimestamp = time.time()
						if evaluate:
							self.evaluate(sess, iteration, self.batch_size)

			except (KeyboardInterrupt, SystemExit):
				self.saveModel(sess, saver, saveModel, currentIteration, TBPreTrainGenIter, TBGenIter, TBDiscIter)

	def play(self):
		tf.reset_default_graph()
		#elf.discriminator = Discriminator(sequence_length=4, vocab_size=6, batch_size=3)
		self.generator = Generator(sequence_length=4, start_token_symbol=0, vocab_size=6, batch_size=3)

		dummyPost = [[0,1,2,3],
					[3,2,1,0],
					[2,2,3,1]]
		dummyReply = dummyPost
		dummyLabels = [1,0,0]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(1000):
				_, result = self.generator.pretrain(sess, dummyPost, dummyReply)
				if i % 10 == 0:
					print(result)
			#result = sess.run(
			#		[self.generator.sequence_logits],
			#		{self.generator.post_seq: dummyPost, self.generator.reply_seq: dummyReply}
			#	)
			#print(result)
			#result = sess.run(
			#	[self.generator.seqences],
			#	{self.generator.post_seq: dummyPost, self.generator.reply_seq: dummyReply})
			#result = self.discriminator.evaluate(sess, dummyPost, dummyReply)
			#print(result)

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
	#GANChat().play()