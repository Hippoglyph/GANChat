import requests
import time
import os
import json
import random
import re
import sys
from TokenProcessor import TokenProcessor
from ProgressPrint import ProgressPrint

pathToReddit = os.path.realpath("Reddit")
pathToDataset = os.path.join(pathToReddit, "dataset")
pathToProcess = os.path.join(pathToReddit, "processed")
pathToProcessed = os.path.join(pathToProcess, "datasetProcessed")
pathToTextFinal = os.path.join(pathToProcess, "datasetTextFinal")
pathToTextTrain = os.path.join(pathToTextFinal, "train")
pathToTextTest = os.path.join(pathToTextFinal, "test")
pathToSequenceDataset = os.path.join(pathToProcess, "sequenceDataset")
pathToSeqTrain = os.path.join(pathToSequenceDataset, "train")
pathToSeqTest = os.path.join(pathToSequenceDataset, "test")

class Processor():
	def __init__(self):
		self.tokenProcessor = TokenProcessor()
		self.dataPointsPerFile = 50000
		self.keepListName = "tokenKeepList.json"
		self.wordToIndexFileName = self.tokenProcessor.wordToIndexFileName
		self.minReplys = 10
		self.maxReplys = 30
		self.keepTopTokens = 30000
		self.minLenKeep = 5
		self.maxLenKeep = 63
		self.unkLimit = 1
		self.testDatasetIdSize = 2000
		random.seed(1337)

		if not os.path.exists(pathToProcess):
			os.makedirs(pathToProcess)

	def processDataset(self):
		print("--Producing process dataset")
		if not os.path.exists(pathToDataset):
			print("No dataset folder")
			return
		if not os.path.exists(pathToProcessed):
			os.makedirs(pathToProcessed)

		datapoints = []
		postCountDict = {}

		print("Replacing files")
		pp = ProgressPrint(len(os.listdir(pathToDataset)))

		removedDatapoints = 0
		storedDataPoints = 0

		for i, fileName in enumerate(os.listdir(pathToDataset)):
			pp.print(i)
			with open(os.path.join(pathToDataset, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					postText = self.tokenProcessor.replaceText(jsonObject["post"])
					replyText = self.tokenProcessor.replaceText(jsonObject["reply"])
					if self.isDatapointCorrectLen(postText, replyText):
						datapoints.append({"post_id": jsonObject["post_id"], "subreddit": jsonObject["subreddit"], "post": postText, "reply": replyText})
						if jsonObject["post_id"] in postCountDict:
							postCountDict[jsonObject["post_id"]] += 1
						else:
							postCountDict[jsonObject["post_id"]] = 1
					else:
						removedDatapoints += 1
		pp.done()
		print("Storing files")
		dataList = []
		fileId = 0
		pp.start(len(datapoints))
		for i, datapoint in enumerate(datapoints):
			pp.print(i)
			if postCountDict[datapoint["post_id"]] >= self.minReplys and postCountDict[datapoint["post_id"]] <= self.maxReplys:
				dataList.append(datapoint)
				storedDataPoints += 1
			else:
				removedDatapoints += 1
			if len(dataList) >= self.dataPointsPerFile:
				self.dumpJson(dataList, fileId, pathToProcessed)
				dataList = []
				fileId += 1
		self.dumpJson(dataList, fileId, pathToProcessed)
		pp.done()
		print("Removed " + str(removedDatapoints) + " datapoints")
		print("Kept " + str(storedDataPoints) + " datapoints")

	def isDatapointCorrectLen(self, post, reply):
		postSplit = self.tokenProcessor.tokenize(post)
		replySplit = self.tokenProcessor.tokenize(reply)
		if len(postSplit) < self.minLenKeep or len(postSplit) > self.maxLenKeep:
			return False
		if len(replySplit) < self.minLenKeep or len(replySplit) > self.maxLenKeep:
			return False
		return True

	def dumpJson(self, dataList, fileID, path):
		if len(dataList) > 0:
			with open(os.path.join(path, str(fileID) + ".json"), 'w', encoding="utf-8") as file:
				for dataPoint in dataList:
					json.dump(dataPoint, file)
					file.write("\n")

	def writeAllProccessedToFile(self):
		if not os.path.exists(pathToProcessed):
			print("No dataset folder")
			return

		pp = ProgressPrint(len(os.listdir(pathToProcessed)))

		datapoints = []

		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					datapoints.append(jsonObject["post"] + " " + jsonObject["reply"])

		with open(os.path.join(pathToReddit, "bigfile"), 'w', encoding="utf-8") as file:
			for item in datapoints:
				file.write(item+ "\n")

		pp.done()

	def writeSampleSplitFile(self):
		if not os.path.exists(pathToProcessed):
			print("No dataset folder")
			return

		pp = ProgressPrint(len(os.listdir(pathToProcessed)))

		datapoints = []

		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					if random.random() < 0.01:
						jsonObject = json.loads(dataPoint)
						datapoints.append(jsonObject["post"])
						datapoints.append(jsonObject["reply"])

		with open(os.path.join(pathToReddit, "sampleSplit"), 'w', encoding="utf-8") as file:
			for item in datapoints:
				json.dump({"from" : item, "to": self.tokenProcessor.tokenize(item)}, file)
				file.write("\n")

		pp.done()

	def wordCountAllFiles(self):
		print("--Word count processed data")
		if not os.path.exists(pathToProcessed):
			print("No dataset folder")
			return

		keepTop = self.keepTopTokens

		tokenMap = {}

		print("Creating token keep list")

		pp = ProgressPrint(len(os.listdir(pathToProcessed)))

		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					for word in self.tokenProcessor.tokenize(jsonObject["post"]) + self.tokenProcessor.tokenize(jsonObject["reply"]):
						if word in tokenMap:
							tokenMap[word] += 1
						else:
							tokenMap[word] = 1

		sortedList = sorted(tokenMap.items(), key = lambda kv: (kv[1], kv[0]), reverse= True)

		accSumOfKeep = sum([item[1] for item in sortedList[:keepTop]])
		accSumOfAll = sum([item[1] for item in sortedList])
		print("If keep top " + str(keepTop) + " tokens " + str(round((accSumOfKeep/accSumOfAll)*100,1)) + "% will be kept" )

		with open(os.path.join(pathToProcess, self.keepListName), 'w', encoding="utf-8") as file:
			for item in sortedList[:keepTop]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")

		pp.done()

	def produceFinalTextDataset(self):
		print("--Producing final text dataset")
		if not os.path.exists(os.path.join(pathToProcess, self.keepListName)):
			print("No keep list")
			return

		if not os.path.exists(pathToTextFinal):
			os.makedirs(pathToTextFinal)
		if not os.path.exists(pathToTextTrain):
			os.makedirs(pathToTextTrain)
		if not os.path.exists(pathToTextTest):
			os.makedirs(pathToTextTest)
			

		print("Constructing token set")
		tokenSet = set()

		with open(os.path.join(pathToProcess, self.keepListName), "r") as keepFile:
			for token in keepFile.readlines():
				tokenSet.add(json.loads(token)["word"])

		datapoints = []
		postCountDict = {}

		removedDatapoints = 0
		storedDataPoints = 0
		postsWithUnknownTokens = 0
		repliesWithUnknownTokens = 0
		
		print("Reading data")
		pp = ProgressPrint(len(os.listdir(pathToProcessed)))
		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					replyUnkCount = 0
					postUnkCount = 0
					for word in self.tokenProcessor.tokenize(jsonObject["reply"]):
						if word not in tokenSet:
							replyUnkCount += 1
					for word in self.tokenProcessor.tokenize(jsonObject["post"]):
						if word not in tokenSet:
							postUnkCount += 1


					if replyUnkCount <= self.unkLimit and postUnkCount <= self.unkLimit:
						datapoints.append({"post_id": jsonObject["post_id"], "subreddit": jsonObject["subreddit"], "post": jsonObject["post"], "reply": jsonObject["reply"]})
						if jsonObject["post_id"] in postCountDict:
							postCountDict[jsonObject["post_id"]] += 1
						else:
							postCountDict[jsonObject["post_id"]] = 1
					else:
						removedDatapoints += 1

		testIdSet = set()
		while len(testIdSet) < self.testDatasetIdSize:
			postID = random.sample(postCountDict.keys(),1)[0]
			if postID not in testIdSet and postCountDict[postID] >= self.minReplys and postCountDict[postID] <= self.maxReplys:
				testIdSet.add(postID)

		dataList = []
		fileId = 0
		dataListTest = []
		fileIdTest = 0
		
		print("Storing data")
		pp.start(len(datapoints))
		for i, datapoint in enumerate(datapoints):
			pp.print(i)
			if postCountDict[datapoint["post_id"]] >= self.minReplys and postCountDict[datapoint["post_id"]] <= self.maxReplys:
				if datapoint["post_id"] in testIdSet:
					dataListTest.append(datapoint)
				else:
					dataList.append(datapoint)
				storedDataPoints += 1
				for word in self.tokenProcessor.tokenize(datapoint["post"]):
					if word not in tokenSet:
						postsWithUnknownTokens += 1
						break
				for word in self.tokenProcessor.tokenize(datapoint["reply"]):
					if word not in tokenSet:
						repliesWithUnknownTokens += 1
						break
			else:
				removedDatapoints += 1
			if len(dataList) >= self.dataPointsPerFile:
				self.dumpJson(dataList, fileId, pathToTextTrain)
				dataList = []
				fileId += 1
			if len(dataListTest) >= self.dataPointsPerFile:
				self.dumpJson(dataListTest, fileIdTest, pathToTextTest)
				dataListTest = []
				fileIdTest += 1
		self.dumpJson(dataList, fileId, pathToTextTrain)
		self.dumpJson(dataListTest, fileIdTest, pathToTextTest)
		pp.done()

		print("Discarded " + str(removedDatapoints) + " dataPoints")
		print("Kept " + str(storedDataPoints) + " dataPoints")
		print("Kept " + str(round((storedDataPoints/(storedDataPoints+removedDatapoints))*100,1)) + '%' + " of the dataset")
		print(str(postsWithUnknownTokens) + " datapoints with unknown tokens in post")
		print(str(repliesWithUnknownTokens) + " datapoints with unknown tokens in reply")

	def countSubredditRatio(self):
		if not os.path.exists(pathToTextFinal):
			print("No dataset")
			return

		subredditsDict = {}
		totalDataPoints = 0
		print("--Calculating subreddit ratio")
		pp = ProgressPrint(len(os.listdir(pathToTextFinal)))
		for i, fileName in enumerate(os.listdir(pathToTextFinal)):
			pp.print(i)
			with open(os.path.join(pathToTextFinal, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					totalDataPoints += 1
					if jsonObject["subreddit"] in subredditsDict:
						subredditsDict[jsonObject["subreddit"]] += 1
					else:
						subredditsDict[jsonObject["subreddit"]] = 1

		for subreddit in subredditsDict:
			print(subreddit + ": " + str(round((subredditsDict[subreddit]/totalDataPoints)*100,1)) + '%')

	def printdatasetInfo(self):
		print("--Printing Information on final text dataset")
		if not os.path.exists(pathToTextFinal):
			print("No dataset folder")
			return
		for path in [pathToTextTrain, pathToTextTest]:
			if not os.path.exists(path):
				print("No sub folder")
				return
			print("Reading...")
			
			tokenSet = set()

			with open(os.path.join(pathToProcess, self.keepListName), "r") as keepFile:
				for token in keepFile.readlines():
					tokenSet.add(json.loads(token)["word"])

			subredditsDict = {}
			postIdCountDict = {}
			wordCountDict = {}
			wordCountDict[self.tokenProcessor.replaceTokens["unknown"]] = 0
			totalDataPoints = 0
			totalPosts = 0
			postLen = 0
			replyLen = 0
			maxLenReply = -sys.maxsize - 1
			minLenReply = sys.maxsize
			maxLenPost = -sys.maxsize - 1
			minLenPost = sys.maxsize
			
			pp = ProgressPrint(len(os.listdir(path)))
			for i, fileName in enumerate(os.listdir(path)):
				pp.print(i)
				with open(os.path.join(path, fileName), "r") as file:
					for dataPoint in file.readlines():
						jsonObject = json.loads(dataPoint)

						totalDataPoints += 1

						if jsonObject["subreddit"] in subredditsDict:
							subredditsDict[jsonObject["subreddit"]] += 1
						else:
							subredditsDict[jsonObject["subreddit"]] = 1

						postSplit = self.tokenProcessor.tokenize(jsonObject["post"])
						replySplit = self.tokenProcessor.tokenize(jsonObject["reply"])

						postLen += len(postSplit)
						replyLen += len(replySplit)
						maxLenReply = max(len(replySplit), maxLenReply)
						maxLenPost = max(len(postSplit), maxLenPost)
						minLenReply = min(len(replySplit), minLenReply)
						minLenPost = min(len(postSplit), minLenPost)

						if jsonObject["post_id"] in postIdCountDict:
							postIdCountDict[jsonObject["post_id"]] += 1
						else:
							postIdCountDict[jsonObject["post_id"]] = 1
							totalPosts += 1

						for word in postSplit + replySplit:
							if word in tokenSet:
								if word in wordCountDict:
									wordCountDict[word] += 1
								else:
									wordCountDict[word] = 0
							else:
								wordCountDict[self.tokenProcessor.replaceTokens["unknown"]] += 1
			pp.done()

			print("Total Datapoints: " + str(totalDataPoints))
			print("Total Posts: " + str(totalPosts))
			print("Replies per post: " + str(round((totalDataPoints/totalPosts),1)) + " avg")
			print("Subbreddit ratio")
			for subreddit in subredditsDict:
				print(subreddit + ": " + str(round(100*(subredditsDict[subreddit]/totalDataPoints),1)) + '%')
			print("Reply length: " + str(round(replyLen/totalDataPoints, 1)) + " avg")
			print("Post length: " + str(round(postLen/totalDataPoints, 1)) + " avg")
			print("Unknown words: " + str(wordCountDict[self.tokenProcessor.replaceTokens["unknown"]]) + " of " + str(sum(wordCountDict.values())))
			print("Unknown words: " + str(round(100*(wordCountDict[self.tokenProcessor.replaceTokens["unknown"]]/(sum(wordCountDict.values()))),1))+ '%')
			print("Max/min length replies: " + str(maxLenReply)+"/"+str(minLenReply))
			print("Max/min length posts: " + str(maxLenPost)+"/"+str(minLenPost))

	def printSequenceDatasetInfo(self):
		print("--Printing Information on sequence dataset")
		if not os.path.exists(pathToSequenceDataset):
			print("No dataset folder")
			return
		for path in [pathToSeqTrain, pathToSeqTest]:
			if not os.path.exists(path):
				print("No sub folder")
				return
			print("Reading...")

			wordCount = 0
			unkCount = 0
			totalDataPoints = 0
			postLen = 0
			replyLen = 0
			
			pp = ProgressPrint(len(os.listdir(path)))
			for i, fileName in enumerate(os.listdir(path)):
				pp.print(i)
				with open(os.path.join(path, fileName), "r") as file:
					for dataPoint in file.readlines():
						jsonObject = json.loads(dataPoint)

						totalDataPoints += 1

						postSeq = jsonObject["post"]
						replySeq = jsonObject["reply"]

						postLen += len([i for i in postSeq if i >= 1])
						replyLen += len([i for i in replySeq if i >= 1])

						for word in postSeq + replySeq:
							if word >= 2:
								wordCount += 1
							if word == 2:
								unkCount += 1
			pp.done()

			print("Total Datapoints: " + str(totalDataPoints))
			print("Reply length: " + str(round(replyLen/totalDataPoints, 1)) + " avg")
			print("Post length: " + str(round(postLen/totalDataPoints, 1)) + " avg")
			print("Unknown words: " + str(unkCount) + " of " + str(wordCount))
			print("Unknown words: " + str(round(100*(unkCount/wordCount),1))+ '%')

	def createSequenceDataset(self):
		print("--Creating sequence dataset")
		if not os.path.exists(pathToTextFinal):
			print("No dataset folder")
			return
		if not os.path.exists(pathToTextTrain):
			print("No sub folder")
			return
		if not os.path.exists(pathToTextTest):
			print("No sub folder")
			return
		if not os.path.exists(pathToSequenceDataset):
			os.makedirs(pathToSequenceDataset)
		if not os.path.exists(pathToSeqTrain):
			os.makedirs(pathToSeqTrain)
		if not os.path.exists(pathToSeqTest):
			os.makedirs(pathToSeqTest)
		print("Reading training...")

		wordCountDict = {}

		trainingDatapoints = []
		testDatapoints = []
		pp = ProgressPrint(len(os.listdir(pathToTextTrain)))
		for i, fileName in enumerate(os.listdir(pathToTextTrain)):
			pp.print(i)
			with open(os.path.join(pathToTextTrain, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					for word in self.tokenProcessor.tokenize(jsonObject["post"])  + self.tokenProcessor.tokenize(jsonObject["reply"]):
						if word in wordCountDict:
							wordCountDict[word] += 1
						else:
							wordCountDict[word] = 1
					trainingDatapoints.append({"post": jsonObject["post"], "reply": jsonObject["reply"]})
		pp.done()

		wordList = sorted(wordCountDict.items(), key = lambda kv: (kv[1], kv[0]), reverse= True)[:self.keepTopTokens]
		wordToIndex = {}
		for i in range(len(wordList)):
			wordToIndex[wordList[i][0]] = i+3
		wordToIndex[self.tokenProcessor.replaceTokens["pad"]] = 0
		wordToIndex[self.tokenProcessor.replaceTokens["start"]] = 1
		wordToIndex[self.tokenProcessor.replaceTokens["unknown"]] = 2

		with open(os.path.join(pathToProcess, self.wordToIndexFileName), 'w', encoding="utf-8") as file:
			for k,v in wordToIndex.items():
				json.dump({"word": k, "id": v}, file)
				file.write("\n")

		print("Reading test...")
		pp = ProgressPrint(len(os.listdir(pathToTextTest)))
		for i, fileName in enumerate(os.listdir(pathToTextTest)):
			pp.print(i)
			with open(os.path.join(pathToTextTest, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					testDatapoints.append({"post": jsonObject["post"], "reply": jsonObject["reply"]})
		pp.done()

		print("Storing data...")
		pp = ProgressPrint(len(trainingDatapoints))

		trainList = []
		testList = []
		trainFileId = 0
		testFileId = 0
		for i,datapoint in enumerate(trainingDatapoints):
			pp.print(i)
			trainList.append({"post": self.tokenProcessor.textToPostSequence(datapoint["post"], self.maxLenKeep), "reply": self.tokenProcessor.textToReplySequence(datapoint["reply"], self.maxLenKeep)})
			if len(trainList) >= self.dataPointsPerFile:
				self.dumpJson(trainList, trainFileId, pathToSeqTrain)
				trainList = []
				trainFileId += 1
		self.dumpJson(trainList, trainFileId, pathToSeqTrain)
		for datapoint in testDatapoints:
			testList.append({"post": self.tokenProcessor.textToPostSequence(datapoint["post"], self.maxLenKeep), "reply": self.tokenProcessor.textToReplySequence(datapoint["reply"], self.maxLenKeep)})
			if len(testList) >= self.dataPointsPerFile:
				self.dumpJson(testList, testFileId, pathToSeqTest)
				testList = []
				testFileId += 1
		self.dumpJson(testList, testFileId, pathToSeqTest)
		pp.done()

def main():
	p = Processor()
	#p.processDataset()
	#p.writeAllProccessedToFile()
	#p.writeSampleSplitFile()
	#p.wordCountAllFiles()
	#p.produceFinalTextDataset()
	#p.printdatasetInfo()
	#p.countSubredditRatio()
	#p.createSequenceDataset()
	p.printSequenceDatasetInfo()


if __name__ == "__main__":
	main()
