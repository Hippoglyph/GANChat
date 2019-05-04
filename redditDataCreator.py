import requests
import time
import os
import json
import random
import re
import unidecode

pathToReddit = os.path.realpath("Reddit")

class Tracker():

	def __init__(self):
		self.throttleTarget = 550
		self.sleepTime = 1
		self.epochReset()

	def epochReset(self):
		self.epochTime = time.time()
		self.requests = 0

	def request(self, url):
		self.throttle()
		self.requests += 1
		#print(url)
		return requests.get(url)

	def epochCheck(self):
		if time.time() - self.epochTime >= 60:
			print("Summary:")
			rpm = self.requests/((time.time() - self.epochTime)/60)
			print(" "+str(int(rpm)) +str(" r/m"))
			self.epochReset()

	def throttle(self):
		self.epochCheck()
		while self.requests > self.throttleTarget:
			time.sleep(self.sleepTime)
			print("Sleep")
			self.epochCheck()

class RedditDataCreator():

	def __init__(self):
		self.commentsPerIteration = 500
		self.minWordCount = 5
		self.maxWordCount = 100
		self.tracker = Tracker()
		self.subreddits = ["politics", "askmen", "askwomen", "jokes", "showerthoughts", "askreddit", "atheism", "worldnews", "dadjokes", "explainlikeimfive"]
		self.replaceTokens = {"url": "xx_URL_Mention_xx"}

	def start(self):
		self.startNewSubmissionPull()

	def startNewSubmissionPull(self):
		newSubreddit = random.choice(self.subreddits)

		if os.path.isfile(os.path.join(pathToReddit, newSubreddit)):
			with open(os.path.join(pathToReddit, newSubreddit), "r") as file:
				before = file.read()
		else:
			before = None

		print("Starting new pull from "+newSubreddit+" from " + (before if before else "start"))

		for submission in self.getSubmissions(newSubreddit, before=before):
			if not before:
				with open(os.path.join(pathToReddit, newSubreddit+"Start"), "w") as file:
					file.write(str(submission["created_utc"]))
			before = submission["created_utc"]

			post = self.submissionQualify(submission)
			if post:
				print("-")
				print(post)
				print("-")

			with open(os.path.join(pathToReddit, newSubreddit), "w") as file:
				file.write(str(before))

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
			post += "\n"+submission["selftext"]+"\n"

		post = self.cleanFromUrls(post.lower())

		wordCount = len(post.split())
		if wordCount >= self.minWordCount and wordCount <= self.maxWordCount:
			return unidecode.unidecode(post)
		return None

	def cleanFromUrls(self, text):
		return re.sub(r'https?\S+', self.replaceTokens["url"], text)

	def getTopComments(self, link_id):
		if not link_id:
			return []

		before = None
		topComments = []

		while True:
			currentComments = self.getComments(link_id, size = self.commentsPerIteration, before=before)

			for comment in currentComments:
				if comment["parent_id"].startswith("t3_"):
					topComments.append(comment)
				before = comment["created_utc"]

			if len(currentComments) < self.commentsPerIteration:
				break

		return topComments


	def getSubmissions(self, subreddit,size=500,sort="desc",sort_type="created_utc",before=None, numCommentsMin=20):
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


		r = self.tracker.request("https://api.pushshift.io/reddit/submission/search/"+params)
		return r.json()["data"]

	def getComments(self, link_id,size=500,sort="desc",sort_type="created_utc",before=None):
		params = "?"

		if link_id:
			params += "link_id="+link_id
		else:
			print("You need to provide a link_id")
			return None

		params += "&size="+str(size)+"&sort="+sort+"&sort_type="+sort_type

		if before:
			params += "&before="+str(before)

		r = self.tracker.request("https://api.pushshift.io/reddit/comment/search/"+params)
		return r.json()["data"]


def main():
	RedditDataCreator().start()

if __name__ == "__main__":
	main()