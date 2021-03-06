import numpy as np
import os
import json
import random

datasetDir = os.path.join(os.path.dirname(__file__), "..", "Reddit", "processed", "sequenceDataset", "train")

class TargetDataLoader():
	def __init__ (self):
		self.data = []
		self.num_batches = 0
		self.pointer = 0

	def appendBatch(self, post, reply):
		self.data.append([post, reply])
		self.num_batches += 1

	def nextBatch(self):
		batch = self.data[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return batch

class RealDataLoader():
	def __init__ (self, batch_size, epochSize):
		self.data = []
		self.num_batches = epochSize//batch_size
		self.pointer = 0
		self.epochSize = epochSize
		self.path = datasetDir
		self.batch_size = batch_size
		self.loadDataset()

	def loadDataset(self):
		print("Loading real dataset...")
		files = os.listdir(self.path)
		for i, fileName in enumerate(files):
			with open(os.path.join(self.path, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					self.data.append([jsonObject["post"],jsonObject["reply"]])
		random.shuffle(self.data)
		self.data = np.array(self.data[:self.num_batches*self.batch_size])
		self.data = np.split(self.data, self.num_batches, 0)

	def nextBatch(self):
		batch = np.transpose(self.data[self.pointer], (1,0,2))
		self.pointer = (self.pointer + 1) % self.num_batches
		return batch

	def getSequenceLength(self):
		return len(self.data[0][0][0])

class GenDataLoader():
	def __init__ (self, batch_size, vocab_size, sequence_length):
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.sequence_length = sequence_length
		self.data = []
		self.num_batches = 0
		self.pointer = 0

	def createDataset(self, model, datapoints, sess, targetDataLoader):
		self.data = []
		self.num_batches = datapoints//self.batch_size

		for _ in range(self.num_batches):
			if targetDataLoader:
				post, _ = targetDataLoader.nextBatch()
			else:
				post = np.random.randint(self.vocab_size, size=(self.batch_size, self.sequence_length))
			self.data.append([post, model.generate(sess, post)])

		self.pointer = 0

	def nextBatch(self):
		post, reply = self.data[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return post, reply

	def getRandomBatch(self):
		return self.data[np.random.randint(self.num_batches)]

class DiscDataLoader():
	def __init__(self, batch_size, vocab_size, sequence_length, trueDataLoader):
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.sequence_length = sequence_length
		self.data = []
		self.num_batches = 0
		self.pointer = 0
		self.trueDataLoader = trueDataLoader

	def createDataset(self, model, datapoints, sess):

		positiveData = []
		negativeData = []

		for post, reply in self.trueDataLoader.data:
			fakereply = model.generate(sess, post)
			for i in range(self.batch_size):
				positiveData.append([post[i], reply[i]])
				negativeData.append([post[i], fakereply[i]])

		self.data = np.array(positiveData + negativeData)
		self.labels = np.array([1 for _ in positiveData] + [0 for _ in negativeData])

		shuffleIndex = np.random.permutation(np.arange(len(self.labels)))

		self.data = self.data[shuffleIndex]
		self.labels = self.labels[shuffleIndex]

		self.num_batches = len(self.labels)//self.batch_size
		self.data = self.data[:self.num_batches*self.batch_size]
		self.labels = self.labels[:self.num_batches*self.batch_size]

		self.data = np.split(self.data, self.num_batches, 0)
		self.labels = np.split(self.labels, self.num_batches, 0)
		self.pointer = 0

	def nextBatch(self):
		post, reply = np.transpose(self.data[self.pointer], (1,0,2))
		label = self.labels[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return post, reply, label


