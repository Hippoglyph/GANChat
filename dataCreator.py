import os
import tarfile
import bz2
import json

pathToTweets = os.path.realpath("Tweets")
pathToRaw = os.path.join(pathToTweets, "RawStreams")
tempbzFolder = os.path.join(pathToTweets, "tmp")
pathToEnglishFiltered = os.path.join(pathToTweets, "EnTweets")

class RawTweetExtractor():

	def __init__(self):
		self.totalTweets = 0
		self.englishTweets = 0
		self.numberOfTweetsPerFile = 100000
		self.fileId = 0
		self.list = []

	def feed(self, tweet):
		if "delete" not in tweet:
			self.totalTweets += 1

			if tweet["lang"] == "en":
				#Filter URLs and images
				self.englishTweets += 1
				self.appendTweet(tweet)
		
	def appendTweet(self, tweet):

		self.list.append({"id": tweet["id"], "in_reply_to_status_id": tweet["in_reply_to_status_id"], "text": tweet["text"]})

		if len(self.list) >= self.numberOfTweetsPerFile:
			self.dumpTweets()

	def dumpTweets(self):
		if len(self.list) > 0:
			#Add instance id token to name
			with open(os.path.join(pathToEnglishFiltered, str(self.fileId) + ".json"), 'w+') as file:
				for tweet in self.list:
					json.dump(tweet, file)
					file.write("\n")

			self.fileId += 1
			self.list = []

def extractTmpbz(rte):
	for file in os.listdir(tempbzFolder):
		fullPath = os.path.join(tempbzFolder, file)
		with bz2.BZ2File(fullPath, "rb") as bzFile:
			for tweet in bzFile.readlines():
				rte.feed(json.loads(tweet.decode("utf8")))
		os.remove(fullPath)

def extractRawTweets():
	if not os.path.exists(pathToEnglishFiltered):
		os.makedirs(pathToEnglishFiltered)
	rte = RawTweetExtractor()
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
							extractTmpbz(rte)
	rte.dumpTweets()
	#Write to file instead
	print("  TotalTweets:" + rte.totalTweets)
	print("EnglishTweets:" + rte.englishTweets)

def main():
	extractRawTweets()
							


if __name__ == "__main__":
	main()