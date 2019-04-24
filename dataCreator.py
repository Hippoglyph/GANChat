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
		self.filteredTweets = 0
		self.numberOfTweetsPerFile = 100000
		self.fileId = 0
		self.list = []

	def feed(self, tweet):
		if "delete" in tweet:
			return
		self.totalTweets += 1

		if tweet["lang"] == "en" and "retweeted_status" not in tweet:
			if tweetContainsLinks(tweet):
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
			#Add instance id token to name
			with open(os.path.join(pathToEnglishFiltered, "A" + str(self.fileId) + ".json"), 'w+') as file:
				for tweet in self.list:
					json.dump(tweet, file)
					file.write("\n")

			self.fileId += 1
			self.list = []

def tweetContainsLinks(tweet):
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
		return tweetContainsLinks(tweet["tweetContainsLinks"])

	return False


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

	with open("RawTweetExtractorRun.txt", "w+") as file:
		file.write("TotalTweets: " + str(rte.totalTweets))
		file.write("FilteredTweets: " + str(rte.filteredTweets))

def main():
	rte = RawTweetExtractor()
	extractRawTweets()

if __name__ == "__main__":
	main()