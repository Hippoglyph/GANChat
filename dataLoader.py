from ProgressPrint import ProgressPrint
import os
import json
import random
import numpy as np

pathToReddit = os.path.realpath("Reddit")
pathToProcess = os.path.join(pathToReddit, "processed")
pathToSequenceDataset = os.path.join(pathToProcess, "sequenceDataset")
pathToSeqTrain = os.path.join(pathToSequenceDataset, "train")
pathToSeqTest = os.path.join(pathToSequenceDataset, "test")

class DataLoader():
	def __init__(self, batchSize, loadOnlyTest = False):
		self.epochNumber = -1
		self.pointer = 0
		self.trainingDatasetSize = 0
		self.testDatasetSize = 0
		self.batchSize = batchSize

		if loadOnlyTest:
			self.initTestDataset()
			self.sequenceLength = len(self.testDataset[0]["post"])
		else:
			self.initDatasets()
			self.newEpoch()
			self.sequenceLength = len(self.trainingDataset[0]["post"])

	def initTestDataset(self):
		self.testDataset = []
		self.testDatasetSize = self.loadDataset(pathToSeqTest, "test", self.testDataset)

	def initTrainingDataset(self):
		self.trainingDataset = []
		self.trainingDatasetSize = self.loadDataset(pathToSeqTrain, "training", self.trainingDataset)

	def initDatasets(self):
		self.initTrainingDataset()
		self.initTestDataset()

	def loadDataset(self, path, name, dataset):
		print("Loading "+name+" dataset...")
		files = os.listdir(path)
		pp = ProgressPrint(len(files))
		for i, fileName in enumerate(files):
			pp.print(i)
			with open(os.path.join(path, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					dataset.append(jsonObject)
		pp.done()
		return len(dataset)

	def newEpoch(self):
		self.pointer = 0
		self.epochNumber += 1
		print("Epoch "+str(self.epochNumber)+" started")
		random.shuffle(self.trainingDataset)

	def nextBatch(self):
		if self.pointer + self.batchSize >= self.trainingDatasetSize:
			self.newEpoch()
		batch = self.trainingDataset[self.pointer:self.pointer+self.batchSize]
		self.pointer += self.batchSize
		post = []
		reply = []
		for datapoint in batch:
			post.append(datapoint["post"])
			reply.append(datapoint["reply"])
		return post, reply

	def getTestBatch(self, size):
		batch = random.sample(self.testDataset, size)
		post = []
		reply = []
		for datapoint in batch:
			post.append(datapoint["post"])
			reply.append(datapoint["reply"])
		return post, reply

	def getSequenceLength(self):
		return self.sequenceLength

	def getEpoch(self):
		return self.epochNumber

	def getEpochProgress(self):
		return self.pointer/self.trainingDatasetSize

class DiscDataLoader():
	def __init__(self, dataLoader, epochSize):
		self.batchSize = dataLoader.batchSize
		self.epochSize = epochSize
		self.dataLoader = dataLoader
		self.numBatches = 0
		self.pointer = 0
		self.data = []

	def createDataset(self, sess, generator):
		positiveData = []
		negativeData = []

		for _ in range(self.epochSize//self.batchSize):
			post, reply = self.dataLoader.nextBatch()
			fakereply = generator.generate(sess, post)
			for i in range(self.batchSize):
				positiveData.append([post[i], reply[i]])
				negativeData.append([post[i], fakereply[i]])

		self.data = np.array(positiveData + negativeData)
		self.labels = np.array([1 for _ in positiveData] + [0 for _ in negativeData])

		shuffleIndex = np.random.permutation(np.arange(len(self.labels)))

		self.data = self.data[shuffleIndex]
		self.labels = self.labels[shuffleIndex]

		self.numBatches = len(self.labels)//self.batchSize
		self.data = self.data[:self.numBatches*self.batchSize]
		self.labels = self.labels[:self.numBatches*self.batchSize]

		self.data = np.split(self.data, self.numBatches, 0)
		self.labels = np.split(self.labels, self.numBatches, 0)
		self.pointer = 0

	def nextBatch(self):
		post, reply = np.transpose(self.data[self.pointer], (1,0,2))
		label = self.labels[self.pointer]
		self.pointer = (self.pointer + 1) % self.numBatches
		return post, reply, label