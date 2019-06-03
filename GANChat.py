import tensorflow as tf
from Embedding import Embedding
from Generator import Generator
from Discriminator import Discriminator
from TokenProcessor import TokenProcessor
from DataLoader import DataLoader
import numpy as np
import os
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.ERROR)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

modelId = "1"
pathToModelsDir = os.path.join(os.path.dirname(__file__), "models")
pathToModelDir = os.path.join(pathToModelsDir, "model"+modelId)
pathToModel = os.path.join(pathToModelDir, "model.ckpt")
tensorboardDir = os.path.join(os.path.dirname(__file__), "tensorboard")
iterationFile = "iteration"

class MODE:
	preTrainGenerator = 0
	preTrainDiscriminator = 1
	adviserialTraining = 2

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

		self.embedding = Embedding(self.vocab_size, self.embedding_size)
		self.generator = Generator(self.embedding, self.sequence_length, self.start_token, self.vocab_size,self.learning_rate, self.batch_size)
		self.discriminator = Discriminator(self.embedding, self.sequence_length, self.start_token, self.learning_rate, self.batch_size)

	def saveModel(self, sess, saver, saveModel, iteration):
		if saveModel:
			print("Saving model...")
			save_path = saver.save(sess, pathToModel)
			with open(os.path.join(pathToModelDir, iterationFile), 'w', encoding="utf-8") as file:
				file.write(str(iteration))
			print("Model saved ("+str(iteration)+"): " + save_path)

	def loadModel(self, sess, saver):
		if not os.path.exists(pathToModelDir):
			print("loadModel: Model does not exit: " + pathToModelDir)
			return
		print("Loading model...")
		saver.restore(sess, pathToModel)
		with open(os.path.join(pathToModelDir, iterationFile), 'r', encoding="utf-8") as file:
			iteration = file.read()
		print("Model loaded (" + str(iteration)+")")
		return int(iteration)

	def train(self):

		trainingMode = MODE.adviserialTraining
		loadModel = False
		saveModel = True

		saver = tf.train.Saver()

		iterationStart = 0
		currentIteration = 0
		with tf.Session() as sess:
			try:
				writer = tf.summary.FileWriter(tensorboardDir, sess.graph)
				if loadModel:
					iterationStart = self.loadModel(sess, saver)
				else:
					print("Initialize new graph")
					sess.run(tf.global_variables_initializer())

				if trainingMode == MODE.adviserialTraining:
					for iteration in range(iterationStart, 9999999):
						currentIteration = iteration
						print("Iternation " + str(currentIteration) + " started")
						#Generator
						for _ in range(1):
							print("Fetch batch")
							postBatch, replyBatch = self.data_loader.nextBatch()
							print("Gen sequence")
							genSequences = self.generator.generate(sess, postBatch)
							print("Score sequence")
							rewards = self.generator.calculateReward(sess, postBatch, genSequences, self.token_sample_rate, self.discriminator)
							print("Train")
							summary = self.generator.train(sess, postBatch, genSequences, rewards)
						writer.add_summary(summary, iteration)
						
						#Discriminator
						for _ in range(5):
							print("Fetch batch")
							postBatch, replyBatch = self.data_loader.nextBatch()
							print("Gen sequence")
							fakeSequences = self.generator.generate(sess, postBatch)
							realSequences = replyBatch

							posts =  np.concatenate([postBatch, postBatch])
							samples = np.concatenate([fakeSequences, realSequences])
							labels = np.concatenate([np.zeros((self.batch_size,)),np.ones((self.batch_size,))])
							print("Train")
							for _ in range(3):
								index = np.random.choice(samples.shape[0], size=(self.batch_size,), replace=False)
								summary = self.discriminator.train(sess, posts[index], samples[index], labels[index])
							writer.add_summary(summary, iteration)	

			except (KeyboardInterrupt, SystemExit):
				self.saveModel(sess, saver, saveModel, currentIteration)
				
				
if __name__ == "__main__":
	GANChat().train()