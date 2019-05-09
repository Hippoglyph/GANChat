import requests
import time
import os
import json
import random
import re
import sys

pathToReddit = os.path.realpath("BackupRedditMulti")
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
		self.replaceTokens = {"url": "xx_url_xx", "user": "xx_user_xx", "sub": "xx_subreddit_xx", "number": "xx_number_xx", "hashtag": "xx_hashtag_xx"}
		#self.splitString = r"([\s\"\(\)\[\]\{\}\,\.\?\!\%\&\:\;\-\=\\\/\*\^]" + "".join(["|"+self.replaceTokens[key] for key in self.replaceTokens])+")"
		self.splitString = r"([^\w']" + "".join(["|"+self.replaceTokens[key] for key in self.replaceTokens])+")"
		self.dataPointsPerFile = 50000
		self.keepListName = "tokenKeepList.json"
		self.minReplys = 10
		self.maxReplys = 30

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

		for i, fileName in enumerate(os.listdir(pathToDataset)):
			pp.print(i)
			with open(os.path.join(pathToDataset, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					#TODO subreddit
					datapoints.append({"post_id": jsonObject["post_id"], "subreddit": "UNKNOWN", "post": self.replaceText(jsonObject["post"]), "reply": self.replaceText(jsonObject["reply"])})
					if jsonObject["post_id"] in postCountDict:
						postCountDict[jsonObject["post_id"]] += 1
					else:
						postCountDict[jsonObject["post_id"]] = 1
		pp.done()
		print("Storing files")
		dataList = []
		fileId = 0
		pp.start(len(datapoints))
		removedDatapoints = 0
		storedDataPoints = 0
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


	def dumpJson(self, dataList, fileID, path):
		if len(dataList) > 0:
			with open(os.path.join(path, str(fileID) + ".json"), 'w', encoding="utf-8") as file:
				for dataPoint in dataList:
					json.dump(dataPoint, file)
					file.write("\n")


	def replaceText(self, text):
		text = text.lower()
		text = re.sub(r"https?[^\s)\]\}]+", self.replaceTokens["url"], text) #URL
		text = re.sub(r"www\.[^\s)\]\}]+", self.replaceTokens["url"], text) #URL
		text = re.sub(r"&amp;", "&", text) #and
		text = re.sub(r"nbsp;", " ", text) #space
		text = re.sub(r"lt;", "<", text) #less
		text = re.sub(r"gt;", ">", text) #large
		text = re.sub(r"le;", "<=", text) #lessEq
		text = re.sub(r"ge;", ">=", text) #largeEq
		text = re.sub(r"[^\S\r\n]+", " ", text)	#Whitespace
		text = re.sub(r"[\r\n]+", "\n", text) #Newlines
		text = re.sub(r"(?<=\W)\/?u\/[^\s)\]\}]+", self.replaceTokens["user"], " " +text)[1:] #User
		text = re.sub(r"(?<=\W)\/?r\/[^\s)\]\}]+", self.replaceTokens["sub"], " " +text)[1:] #Subreddit
		text = re.sub(r"#\w+", self.replaceTokens["hashtag"], text) #Hashtag


		text = re.sub(r"(?<=\s)lpts?\s?\:?\-?\??\!?('s)?;?", "", " " +text)[1:] #lpt
		text = re.sub(r"(?<=\s)eli5s?\s?\:?\-?\??\!?('s)?;?", "", " " +text)[1:] #eli5

		text = re.sub(r"\d+", self.replaceTokens["number"], text) #numbers
		
		text = re.sub(r"xx_url_mention_xx", self.replaceTokens["url"], text) #LEGACY
		return text

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
				json.dump({"from" : item, "to": self.splitText(item)}, file)
				file.write("\n")

		pp.done()

	def splitText(self, text):
		return list(filter(None,re.split(self.splitString,text)))

	def wordCountAllFiles(self):
		if not os.path.exists(pathToProcessed):
			print("No dataset folder")
			return

		keepTop = 30000

		tokenMap = {}

		print("Creating token keep list")

		pp = ProgressPrint(len(os.listdir(pathToProcessed)))

		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					for word in self.splitText(jsonObject["post"]) + self.splitText(jsonObject["reply"]):
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
				json.dump({"word": item[0]}, file)
				file.write("\n")

		pp.done()

	def removeAllRepliesWithUnknownTokens(self):
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
		
		print("Reading data")
		pp = ProgressPrint(len(os.listdir(pathToProcessed)))
		for i, fileName in enumerate(os.listdir(pathToProcessed)):
			pp.print(i)
			with open(os.path.join(pathToProcessed, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					discard = False
					for word in self.splitText(jsonObject["reply"]):
						if word not in tokenSet:
							discard = True
							break
					if not discard:
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
				for word in self.splitText(datapoint["post"]):
					if word not in tokenSet:
						postsWithUnknownTokens += 1
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
						
		pp.done()

def main():
	#Processor().processDataset()
	#Processor().writeAllProccessedToFile()
	#Processor().writeSampleSplitFile()
	#Processor().wordCountAllFiles()
	Processor().removeAllRepliesWithUnknownTokens()

if __name__ == "__main__":
	main()
