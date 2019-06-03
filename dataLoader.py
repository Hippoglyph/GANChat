from ProgressPrint import ProgressPrint
import os
import json
import random

pathToReddit = os.path.realpath("Reddit")
pathToProcess = os.path.join(pathToReddit, "processed")
pathToSequenceDataset = os.path.join(pathToProcess, "sequenceDataset")
pathToSeqTrain = os.path.join(pathToSequenceDataset, "train")
pathToSeqTest = os.path.join(pathToSequenceDataset, "test")

class DataLoader():
	def __init__(self, batchSize):
		self.epochNumber = -1
		self.pointer = 0
		self.trainingDatasetSize = 0
		self.testDatasetSize = 0
		self.batchSize = batchSize
		self.initDatasets()
		self.newEpoch()
		self.sequenceLength = len(self.trainingDataset[0]["post"])

	def initDatasets(self):
		self.trainingDataset = []
		self.testDataset = []
		self.trainingDatasetSize = self.loadDataset(pathToSeqTrain, "training", self.trainingDataset)
		self.testDatasetSize = self.loadDataset(pathToSeqTest, "test", self.testDataset)

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
		batch = random.sample(testDatasetSize, size)
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