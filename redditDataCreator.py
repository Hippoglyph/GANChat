import requests
import time
import os
import json

pathToReddit = os.path.realpath("Reddit")

class RedditDataCreator():

	def __init__(self):
		self.commentsPerIteration = 500
		pass

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


		r = requests.get("https://api.pushshift.io/reddit/submission/search/"+params)
		print("https://api.pushshift.io/reddit/submission/search/"+params)
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

		r = requests.get("https://api.pushshift.io/reddit/comment/search/"+params)
		print("https://api.pushshift.io/reddit/comment/search/"+params)
		return r.json()["data"]


def main():
	rdc = RedditDataCreator()

	data = rdc.getSubmissions("showerthoughts", size=1, numCommentsMin=600)
	if data:
		coms = rdc.getTopComments(data[0]["id"])
		print(data[0]["url"])
		for com in coms:
			print(com["body"][:50])
		#
		#with open(os.path.join(pathToReddit,"test.json"), 'w+', encoding="utf-8") as file:
		#	for post in data:
		#		json.dump(post, file)
		#		file.write("\n")


if __name__ == "__main__":
	main()