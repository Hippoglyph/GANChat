import os
import tarfile
import bz2
import json
import unidecode
import time
import random
import sys

pathToTweets = os.path.realpath("Tweets")
pathToRaw = os.path.join(pathToTweets, "RawStreams")
tempbzFolder = os.path.join(pathToTweets, "tmp")
pathToEnglishFiltered = os.path.join(pathToTweets, "EnTweets")
pathToError = os.path.join(pathToTweets, "Errors")
pathToRankFilterLists = os.path.join(pathToTweets, "rankFilter")
rawTweetMetaFileName = "RawTweetExtractorRun"
rankFilterMetaFileName = "rankFilterMeta"

class ProgressPrint():

	def __init__(self):
		self.total = 0
		self.lastPrecentage = 0
		

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

class RawTweetExtractor():

	def __init__(self, instanceID):
		self.totalTweets = 0
		self.filteredTweets = 0
		self.numberOfTweetsPerFile = 100000
		self.fileId = 0
		self.list = []
		self.instanceID = instanceID

	def start(self):
		self.extractRawTweets()

	def feed(self, tweet):
		if "delete" in tweet:
			return
		self.totalTweets += 1

		if tweet["lang"] == "en" and "retweeted_status" not in tweet:
			if self.tweetContainsLinks(tweet):
				return
			self.filteredTweets += 1
			self.appendTweet(tweet)
	
	def appendTweet(self, tweet):

		text = tweet["extended_tweet"]["full_text"] if "extended_tweet" in tweet else tweet["text"]

		self.list.append({"id": tweet["id"], "username": tweet["user"]["screen_name"], "in_reply_to_status_id": tweet["in_reply_to_status_id"], "text": text})

		if len(self.list) >= self.numberOfTweetsPerFile:
			self.dumpTweets()

	def dumpTweets(self):
		if len(self.list) > 0:
			with open(os.path.join(pathToEnglishFiltered, self.instanceID + str(self.fileId) + ".json"), 'w+', encoding="utf-8") as file:
				for tweet in self.list:
					json.dump(tweet, file)
					file.write("\n")

			self.fileId += 1
			self.list = []

	def tweetContainsLinks(self, tweet):
		if "entities" not in tweet:
			return False

		if "media" in tweet["entities"] and tweet["entities"]["media"]:
			return True

		if "urls" in tweet["entities"] and tweet["entities"]["urls"]:
			return True

		if "polls" in tweet["entities"] and tweet["entities"]["polls"]:
			return True

		if "extended_entities" in tweet and "media" in tweet["extended_entities"] and tweet["extended_entities"]["media"]:
			return True

		if "extended_tweet" in tweet:
			return self.tweetContainsLinks(tweet["tweetContainsLinks"])

		return False


	def extractTmpbz(self):
		for file in os.listdir(tempbzFolder):
			fullPath = os.path.join(tempbzFolder, file)
			with bz2.BZ2File(fullPath, "rb") as bzFile:
				for tweet in bzFile.readlines():
					if tweet:
						try:
							jsonObject = json.loads(tweet.decode("utf-8", "ignore"))
						except:
							errorid = random.randrange(99999)
							print("Could not proccess tweet ("+str(errorid)+")")
							with open(os.path.join(pathToError, str(errorid)+ ".txt"), "w+", encoding="utf-8") as file:
								file.write(tweet.decode("utf-8", "ignore"))
							continue
						self.feed(jsonObject)
			os.remove(fullPath)

	def extractRawTweets(self):
		start = time.time()
		if not os.path.exists(pathToEnglishFiltered):
			os.makedirs(pathToEnglishFiltered)
		if os.path.exists(tempbzFolder):
			for file in os.listdir(tempbzFolder):
				os.remove(os.path.join(tempbzFolder, file))
		for stream in os.listdir(pathToRaw):
			print("Stream: " + stream)
			pathToFile = os.path.join(pathToRaw, stream)
			for file in os.listdir(pathToFile):
				if file.endswith(".tar"):
					print("------ID: " + file)
					with tarfile.open(os.path.join(pathToFile, file)) as theTarFile:
						for member in theTarFile.getmembers():
							if member.path.endswith(".bz2"):
								member.name = os.path.basename(member.name)
								theTarFile.extract(member, tempbzFolder)
								self.extractTmpbz()
		self.dumpTweets()

		with open(rawTweetMetaFileName+self.instanceID+".txt", "w+") as file:
			file.write("Time: " +str((time.time() - start)/60) + " minutes\n")
			file.write("TotalTweets: " + str(self.totalTweets)+"\n")
			file.write("FilteredTweets: " + str(self.filteredTweets)+"\n")

class rankFilteringExtractor():

	def __init__(self):
		self.replyThreshold = 20
		self.tweetsPerFile = 1000
		self.fileID = 0
		self.idList = []
		self.dictionary = {}
		self.startTime = 0

	def start(self):
		self.startTime = time.time()
		if not os.path.exists(pathToEnglishFiltered):
			print("No raw filtered tweets exits")
			return
		if not os.path.exists(pathToRankFilterLists):
			os.makedirs(pathToRankFilterLists)

		print("Start Counting...")
		self.countReplies()
		print("Present checking...")
		self.presentCheck()
		print("Thresholding...")
		self.removeInvalids()
		print("Writing Result to file...")
		self.writeResult()
		print("Done")

	def countReplies(self):
		pp = ProgressPrint()
		fileList = os.listdir(pathToEnglishFiltered)
		pp.start(len(fileList))
		fileCount = 0
		for fileName in fileList:
			pp.print(fileCount)
			fileCount += 1
			with open(os.path.join(pathToEnglishFiltered, fileName)) as file:
				for tweet in file.readlines():
					if tweet:
						try:
							jsonObject = json.loads(tweet)
						except:
							errorid = random.randrange(99999)
							print("Could not proccess tweet ("+str(errorid)+")")
							with open(os.path.join(pathToError, str(errorid)+ ".txt"), "w+", encoding="utf-8") as fileError:
								fileError.write(tweet)
							continue
						self.feed(jsonObject)
		pp.done()

	def feed(self, tweet):
		#TODO: Check quality of reply
		if tweet["in_reply_to_status_id"]:
			if tweet["in_reply_to_status_id"] in self.dictionary:
				self.dictionary[tweet["in_reply_to_status_id"]]["replayCount"] += 1
			else:
				self.dictionary[tweet["in_reply_to_status_id"]] = {"replayCount": 1, "present": False}

	def presentCheck(self):
		pp = ProgressPrint()
		fileList = os.listdir(pathToEnglishFiltered)
		pp.start(len(fileList))
		fileCount = 0
		for fileName in fileList:
			pp.print(fileCount)
			fileCount += 1
			with open(os.path.join(pathToEnglishFiltered, fileName)) as file:
				for tweet in file.readlines():
					if tweet:
						try:
							jsonObject = json.loads(tweet)
						except:
							errorid = random.randrange(99999)
							print("Could not proccess tweet ("+str(errorid)+")")
							with open(os.path.join(pathToError, str(errorid)+ ".txt"), "w+", encoding="utf-8") as fileError:
								fileError.write(tweet)
							continue
						self.presentFeed(jsonObject)
		pp.done()

	def presentFeed(self, tweet):
		if tweet["id"] in self.dictionary:
			self.dictionary[tweet["id"]]["present"] = True

	def removeInvalids(self):
		self.dictionary = {key:self.dictionary[key]["replayCount"] for key in self.dictionary if self.dictionary[key]["present"] == True and self.dictionary[key]["replayCount"] >= self.replyThreshold}

	def writeResult(self):
		pp = ProgressPrint()
		pp.start(len(self.dictionary))
		keyItr = 0
		totalReplies = 0
		for key in self.dictionary:
			pp.print(keyItr)
			keyItr += 1
			self.idList.append({"id":key,"replayCount": self.dictionary[key]})
			totalReplies += self.dictionary[key]
			if len(self.idList) >= self.tweetsPerFile:
				self.dumpJson()
		self.dumpJson()
		pp.done()

		with open(rankFilterMetaFileName+".txt", "w+") as file:
			file.write("Time: " +str((time.time() - self.startTime)/60) + " minutes\n")
			file.write("TotalPosts: " + str(len(self.dictionary))+"\n")
			file.write("TotalReplies: " + str(totalReplies)+"\n")

	def dumpJson(self):
		if len(self.idList) > 0:
			with open(os.path.join(pathToRankFilterLists, str(self.fileID) + ".json"), 'w+', encoding="utf-8") as file:
				for tweet in self.idList:
					json.dump(tweet, file)
					file.write("\n")

			self.fileID += 1
			self.idList = []

def main():
	#RawTweetExtractor("B").start()
	rankFilteringExtractor().start()

if __name__ == "__main__":
	main()