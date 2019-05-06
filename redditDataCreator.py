import requests
import time
import os
import json
import random
import re
import unidecode
import sys

pathToReddit = os.path.realpath("Reddit")
pathToSubreddits = os.path.join(pathToReddit, "Subreddits")
pathToDataset = os.path.join(pathToReddit, "dataset")
pathToInstanceLogs = os.path.join(pathToReddit, "InstanceLogs")

class Tracker():

	def __init__(self):
		self.throttleTarget = 550
		self.sleepTime = 1
		self.fileId = 0
		self.dataPointsPerFile = 2000
		self.dataPoints = []
		self.startTime = time.time()
		self.totalDataPointsAdded = 0
		self.totalPostsAdded = 0
		self.postId = ""
		self.getInstanceId()
		self.epochReset()

	def getInstanceId(self):
		if not os.path.exists(pathToDataset):
			os.makedirs(pathToDataset)
		if not os.path.exists(pathToInstanceLogs):
			os.makedirs(pathToInstanceLogs)
		logList = os.listdir(pathToInstanceLogs)
		self.instanceId = len(logList)
		with open(os.path.join(pathToInstanceLogs, str(self.instanceId)), "w") as file:
			file.write(str(time.time()))

	def epochReset(self):
		self.epochTime = time.time()
		self.requests = 0
		self.sleepCounter = 0
		self.dataPointsThisMinute = 0

	def request(self, url):
		self.throttle()
		self.requests += 1
		#print(url)
		r = requests.get(url)
		try:
			json = r.json()
		except Exception as e:
			print(e)
			print(r)
			print("Request Error")
			sys.exit()
		return json

	def epochCheck(self):
		if time.time() - self.epochTime >= 60:
			print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
			hoursSinceStart = (time.time() - self.startTime)/(60*60)
			print(" " + str(int(hoursSinceStart)) + " hours since start of session")
			rpm = self.requests/((time.time() - self.epochTime)/60)
			print(" "+str(int(rpm)) +str(" r/m"))
			print(" Added " + str(self.dataPointsThisMinute) + " data points this minute")
			print(" Added " + str(self.totalDataPointsAdded) + " data points this session")
			print(" Added " + str(self.totalPostsAdded) + " posts this session")
			dppm = self.totalDataPointsAdded/((time.time() - self.startTime)/60)
			print(" Adding " + str(int(dppm)) + " dp/m on average")
			if self.sleepCounter > 0:
				print(" Slept for " + str(int(self.sleepCounter*self.sleepTime)) + " seconds this minute")
			print()
			self.epochReset()

	def throttle(self):
		self.epochCheck()
		while self.requests > self.throttleTarget:
			time.sleep(self.sleepTime)
			self.sleepCounter += 1
			self.epochCheck()

	def dumpDataPoints(self):
		if len(self.dataPoints) > 0:
			with open(os.path.join(pathToDataset, str(self.instanceId) + "-" + str(self.fileId) + ".json"), 'w', encoding="utf-8") as file:
				for dataPoint in self.dataPoints:
					json.dump(dataPoint, file)
					file.write("\n")

			self.fileId += 1
			self.dataPoints = []
			print("Stored data points " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

	def appendDataPoint(self,subId, post, reply, subreddit):
		self.dataPoints.append({"post_id": subId, "subreddit": subreddit, "post": post, "reply": reply})

		if subId != self.postId:
			self.postId = subId
			self.totalPostsAdded += 1

		self.dataPointsThisMinute += 1
		self.totalDataPointsAdded += 1

		if len(self.dataPoints) > self.dataPointsPerFile:
			self.dumpDataPoints()

class RedditDataCreator():

	def __init__(self):
		self.commentsPerIteration = 500
		self.minWordCount = 5
		self.maxWordCount = 100
		self.minReplys = 10
		self.maxReplys = 30
		self.hardCapTopComments = 500
		self.submissionsPerPull = 100
		self.tracker = Tracker()
		self.subreddits = ["politics", "jokes", "showerthoughts", "askreddit", "worldnews", "dadjokes", "explainlikeimfive", "lifeprotips", "nostupidquestions", "news", "science", "answers", "askscience"]
		self.replaceTokens = {"url": "xx_url_xx", "user": "xx_user_xx", "sub": "xx_subreddit_xx"}
		self.createFolders()

	def createFolders(self):
		if not os.path.exists(pathToReddit):
			os.makedirs(pathToReddit)
		if not os.path.exists(pathToSubreddits):
			os.makedirs(pathToSubreddits)

	def start(self):
		while True:
			self.startNewSubmissionPull()

	def startNewSubmissionPull(self):
		newSubreddit = random.choice(self.subreddits)

		if os.path.isfile(os.path.join(pathToSubreddits, newSubreddit)):
			with open(os.path.join(pathToSubreddits, newSubreddit), "r") as file:
				before = file.read()
		else:
			before = None

		print("Pulling from "+newSubreddit+" before " + (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(before))) if before else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

		submissions = self.getSubmissions(newSubreddit,size=self.submissionsPerPull, before=before)

		if len(submissions) == 0:
			print("Subreddit: " + newSubreddit + " depleted")
			with open(os.path.join(pathToSubreddits, newSubreddit+"End"), "w") as file:
					file.write("Dead subreddit")
			self.subreddits.remove(newSubreddit)

		for submission in submissions:
			if not before:
				with open(os.path.join(pathToSubreddits, newSubreddit+"Start"), "w") as file:
					file.write(str(submission["created_utc"]))
			before = submission["created_utc"]

			with open(os.path.join(pathToSubreddits, newSubreddit), "w") as file:
				file.write(str(before))

			post = self.submissionQualify(submission)
			if post:
				self.createDataFromSubmission(submission, post)

	def createDataFromSubmission(self, submission, post):
		rawComments = self.getTopComments(submission["id"])
		comments = self.cleanComments(rawComments)
		if len(comments) < self.minReplys:
			return

		if len(comments) > self.maxReplys:
			comments = random.sample(comments, self.maxReplys)

		for comment in comments:
			self.tracker.appendDataPoint(submission["id"],post,comment, submission["subreddit"])

	def cleanComments(self, rawComments):
		comments = []
		for dirtyComment in rawComments:
			if "is_submitter" in dirtyComment and dirtyComment["is_submitter"]:
				continue
			if dirtyComment["body"] == "[removed]" or dirtyComment["body"] == "[deleted]":
				continue
			reply = self.cleanFromUrls(dirtyComment["body"].lower())
			reply = self.removeWhitespace(reply)
			reply = self.removeUser(reply)
			reply = self.removeSub(reply)
			reply = self.removeRepetedNewlines(reply)
			wordCount = len(reply.split())
			if wordCount >= self.minWordCount and wordCount <= self.maxWordCount:
				comments.append(unidecode.unidecode(reply))
		return comments

	def submissionQualify(self, submission):
		if "over_18" in submission and submission["over_18"]:
			return None
		if "media_only" in submission and submission["media_only"]:
			return None
		if "is_video" in submission and submission["is_video"]:
			return None

		post = ""

		if "title" in submission and submission["title"]:
			post += submission["title"]+"\n"
		if "selftext" in submission and submission["selftext"]:
			if submission["selftext"] == "[removed]" or submission["selftext"] == "[deleted]":
				return None
			post += submission["selftext"]+"\n"

		post = self.cleanFromUrls(post.lower())
		post = self.removeWhitespace(post)
		post = self.removeUser(post)
		post = self.removeSub(post)
		post = self.removeRepetedNewlines(post)

		if "url" in submission and not submission["url"].startswith("https://www.reddit.com"):
			post += self.replaceTokens["url"]+"\n"

		wordCount = len(post.split())
		if wordCount >= self.minWordCount and wordCount <= self.maxWordCount:
			return unidecode.unidecode(post)
		return None

	def cleanFromUrls(self, text):
		text = re.sub(r"https?[^\s)\]\}]+", self.replaceTokens["url"], text)
		return re.sub(r"www\.[^\s)\]\}]+", self.replaceTokens["url"], text)

	def removeWhitespace(self, text):
		return re.sub(r"[^\S\r\n]+", " ", text)

	def removeUser(self, text):
		return re.sub(r"(?<=\W)\/?u\/[^\s)]+", self.replaceTokens["user"], " " +text)[1:]

	def removeSub(self, text):
		return re.sub(r"(?<=\W)\/?r\/[^\s)]+", self.replaceTokens["sub"], " " +text)[1:]

	def removeRepetedNewlines(self, text):
		return re.sub(r"[\r\n]+", "\n", text)

	def getTopComments(self, link_id):
		if not link_id:
			return []

		after = None
		topComments = []

		while True:
			currentComments = self.getComments(link_id, size = self.commentsPerIteration, after=after)

			for comment in currentComments:
				if comment["parent_id"].startswith("t3_"):
					topComments.append(comment)
				after = comment["created_utc"]

			if len(currentComments) < self.commentsPerIteration:
				break

			if len(topComments) >= self.hardCapTopComments:
				return topComments

		return topComments


	def getSubmissions(self, subreddit,size=500,sort="desc",sort_type="created_utc",before=None, numCommentsMin=30):
		params = "?"

		if subreddit:
			params += "subreddit="+subreddit
		else:
			print("You need to provide a subreddit")
			return None

		params += "&size="+str(size)+"&sort="+sort+"&sort_type="+sort_type

		if before:
			params += "&before="+str(before)
		if numCommentsMin:
			params += "&num_comments=>"+str(numCommentsMin)


		json = self.tracker.request("https://api.pushshift.io/reddit/submission/search/"+params)
		return json["data"]

	def getComments(self, link_id,size=500,sort="asc",sort_type="created_utc",after=None):
		params = "?"

		if link_id:
			params += "link_id="+link_id
		else:
			print("You need to provide a link_id")
			return None

		params += "&size="+str(size)+"&sort="+sort+"&sort_type="+sort_type

		if after:
			params += "&after="+str(after)

		json = self.tracker.request("https://api.pushshift.io/reddit/comment/search/"+params)
		return json["data"]


def main():
	RedditDataCreator().start()

if __name__ == "__main__":
	main()