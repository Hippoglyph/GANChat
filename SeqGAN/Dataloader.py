import numpy as np

class GenDataLoader():
	def __init__ (self, batch_size):
		self.batch_size = batch_size
		self.data = []
		self.num_batches = 0
		self.pointer = 0

	def createDataset(self, model, datapoints, sess):
		self.data = []
		self.num_batches = datapoints//self.batch_size

		for _ in range(self.num_batches):
			self.data.append(model.generate(sess))

		self.pointer = 0

	def nextBatch(self):
		batch = self.data[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return batch

class DiscDataLoader():
	def __init__(self, batch_size, trueDataLoader):
		self.batch_size = batch_size
		self.data = []
		self.num_batches = 0
		self.pointer = 0
		self.trueDataLoader = trueDataLoader

	def createDataset(self, model, datapoints, sess):

		positiveData = []
		negativeData = []

		for batch in self.trueDataLoader.data:
			for item in batch:
				positiveData.append(item)

		for _ in range(datapoints//self.batch_size):
			batch = model.generate(sess)
			for item in batch:
				negativeData.append(item)

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
		batch = self.data[self.pointer]
		label = self.labels[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batches
		return batch, label


