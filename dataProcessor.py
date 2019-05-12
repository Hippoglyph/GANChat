import requests
import time
import os
import json
import random
import re
import sys
from tokenProcessor import tokenProcessor

pathToReddit = os.path.realpath("Reddit")
pathToDataset = os.path.join(pathToReddit, "dataset")
pathToProcessed = os.path.join(pathToReddit, "dataPostProcessed")
pathToTextFinal = os.path.join(pathToReddit, "datasetTextFinal")

class ProgressPrint():

	def __init__(self, total):
		self.total = 0
		self.lastPrecentage = 0
		self.start(total)

	def start(self, total):
		self.total = total
		self.write(0)

	def done(self):
		self.write(100)
		print("")

	def print(self, number):
		procentage = int((100*(number/self.total)))
		if procentage != self.lastPrecentage:
			self.lastPrecentage = procentage
			self.write(self.lastPrecentage)

	def write(self, number):
		print("{:>3}%".format(number),end="\r")

class Processor():
	def __init__(self):
		self.tokenProcessor = tokenProcessor()
		self.dataPointsPerFile = 50000
		self.keepListName = "tokenKeepList.json"
		self.minReplys = 10
		self.maxReplys = 30
		self.keepTopTokens = 30000
		self.minLenKeep = 5
		self.maxLenKeep = 150

	def processDataset(self):
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
		postSplit = self.tokenProcessor.splitText(post)
		replySplit = self.tokenProcessor.splitText(reply)
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
				json.dump({"from" : item, "to": self.tokenProcessor.splitText(item)}, file)
				file.write("\n")

		pp.done()

	def wordCountAllFiles(self):
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
					for word in self.tokenProcessor.splitText(jsonObject["post"]) + self.tokenProcessor.splitText(jsonObject["reply"]):
						if word in tokenMap:
							tokenMap[word] += 1
						else:
							tokenMap[word] = 1

		sortedList = sorted(tokenMap.items(), key = lambda kv: (kv[1], kv[0]), reverse= True)

		accSumOfKeep = sum([item[1] for item in sortedList[:keepTop]])
		accSumOfAll = sum([item[1] for item in sortedList])
		print("If keep top " + str(keepTop) + " tokens " + str(round((accSumOfKeep/accSumOfAll)*100,1)) + "% will be kept" )
		'''
		with open(os.path.join(pathToReddit, "tokenKeepCount.json"), 'w', encoding="utf-8") as file:
			for item in sortedList[:keepTop]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")
		with open(os.path.join(pathToReddit, "tokenLostCount.json"), 'w', encoding="utf-8") as file:
			for item in sortedList[keepTop:]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")
		'''
		with open(os.path.join(pathToReddit, self.keepListName), 'w', encoding="utf-8") as file:
			for item in sortedList[:keepTop]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")

		pp.done()

	def produceFinalTextDataset(self):
		if not os.path.exists(os.path.join(pathToReddit, self.keepListName)):
			print("No keep list")
			return

		if not os.path.exists(pathToTextFinal):
			os.makedirs(pathToTextFinal)

		print("Constructing token set")
		tokenSet = set()

		with open(os.path.join(pathToReddit, self.keepListName), "r") as keepFile:
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
					for word in self.tokenProcessor.splitText(jsonObject["reply"]):
						if word not in tokenSet:
							replyUnkCount += 1
					for word in self.tokenProcessor.splitText(jsonObject["post"]):
						if word not in tokenSet:
							postUnkCount += 1


					if replyUnkCount <= 1 and postUnkCount <= 1:
						datapoints.append({"post_id": jsonObject["post_id"], "subreddit": jsonObject["subreddit"], "post": jsonObject["post"], "reply": jsonObject["reply"]})
						if jsonObject["post_id"] in postCountDict:
							postCountDict[jsonObject["post_id"]] += 1
						else:
							postCountDict[jsonObject["post_id"]] = 1
					else:
						removedDatapoints += 1

		print("Storing data")
		pp.start(len(datapoints))
		dataList = []
		fileId = 0
		for i, datapoint in enumerate(datapoints):
			pp.print(i)
			if postCountDict[datapoint["post_id"]] >= self.minReplys and postCountDict[datapoint["post_id"]] <= self.maxReplys:
				dataList.append(datapoint)
				storedDataPoints += 1
				for word in self.tokenProcessor.splitText(datapoint["post"]):
					if word not in tokenSet:
						postsWithUnknownTokens += 1
						break
				for word in self.tokenProcessor.splitText(datapoint["reply"]):
					if word not in tokenSet:
						repliesWithUnknownTokens += 1
						break
			else:
				removedDatapoints += 1
			if len(dataList) >= self.dataPointsPerFile:
				self.dumpJson(dataList, fileId, pathToTextFinal)
				dataList = []
				fileId += 1
		self.dumpJson(dataList, fileId, pathToTextFinal)
		pp.done()

		print("Discarded " + str(removedDatapoints) + " dataPoints")
		print("Kept " + str(storedDataPoints) + " dataPoints")
		print("Kept " + str(round((storedDataPoints/(storedDataPoints+removedDatapoints))*100,1)) + '%' + " of the dataset")
		print(str(postsWithUnknownTokens) + " datapoints with unknowned tokens in post")
		print(str(repliesWithUnknownTokens) + " datapoints with unknowned tokens in reply")
						
		pp.done()

	def countSubredditRatio(self):
		if not os.path.exists(pathToTextFinal):
			print("No dataset")
			return

		subredditsDict = {}
		totalDataPoints = 0
		print("Calculating subreddit ratio")
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
		if not os.path.exists(pathToTextFinal):
			print("No dataset folder")
			return
		print("Printing Information")
		print("Reading...")
		pp = ProgressPrint(len(os.listdir(pathToTextFinal)))

		tokenSet = set()

		with open(os.path.join(pathToReddit, self.keepListName), "r") as keepFile:
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

					postSplit = self.tokenProcessor.splitText(jsonObject["post"])
					replySplit = self.tokenProcessor.splitText(jsonObject["reply"])

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

					for word in postSplit:
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

def main():
	p = Processor()
	p.processDataset()
	#p.writeAllProccessedToFile()
	#p.writeSampleSplitFile()
	p.wordCountAllFiles()
	p.produceFinalTextDataset()
	p.printdatasetInfo()
	#p.countSubredditRatio()


if __name__ == "__main__":
	main()
