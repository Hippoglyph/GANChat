import os
import tarfile
import bz2
import json
import unidecode
import time
import random

pathToTweets = os.path.realpath("Tweets")
pathToRaw = os.path.join(pathToTweets, "RawStreams")
tempbzFolder = os.path.join(pathToTweets, "tmp")
pathToEnglishFiltered = os.path.join(pathToTweets, "EnTweets")
pathToError = os.path.join(pathToTweets, "Errors")

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

		with open("RawTweetExtractorRun"+self.instanceID+".txt", "w+") as file:
			file.write("Time: " +str((time.time() - start)/60) + " minutes")
			file.write("TotalTweets: " + str(self.totalTweets))
			file.write("FilteredTweets: " + str(self.filteredTweets))

def main():
	RawTweetExtractor("A").start()

if __name__ == "__main__":
	main()