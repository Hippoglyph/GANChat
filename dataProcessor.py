import requests
import time
import os
import json
import random
import re
import sys

pathToReddit = os.path.realpath("BackupRedditMulti")
pathToDataset = os.path.join(pathToReddit, "dataset")

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

class Proccessor():
	def __init__(self):
		self.splitString = r"([\s\"\(\)\[\]\{\}\?\!\%\&]|(?<!\d)[,.]|[,.](?!\d))"

	def wordCountAllFiles(self):
		if not os.path.exists(pathToDataset):
			print("No dataset folder")
			return

		totalVocab = 0
		keepTop = 30000

		tokenMap = {}

		pp = ProgressPrint()
		pp.start(len(os.listdir(pathToDataset)))

		fileCount = 0
		for fileName in os.listdir(pathToDataset):
			pp.print(fileCount)
			fileCount += 1
			with open(os.path.join(pathToDataset, fileName), "r") as file:
				for dataPoint in file.readlines():
					jsonObject = json.loads(dataPoint)
					for word in list(filter(None,re.split(self.splitString,jsonObject["post"]))) + list(filter(None,re.split(self.splitString,jsonObject["reply"]))):
						if word in tokenMap:
							tokenMap[word] += 1
						else:
							tokenMap[word] = 1
							totalVocab += 1

		sortedList = sorted(tokenMap.items(), key = lambda kv: (kv[1], kv[0]), reverse= True)

		accSumOfKeep = sum([item[1] for item in sortedList[:keepTop]])
		accSumOfAll = sum([item[1] for item in sortedList])
		print("If keep top " + str(keepTop) + " tokens " + str(round((accSumOfKeep/accSumOfAll)*100,1)) + "% will be kept" )

		with open(os.path.join(pathToReddit, "tokenKeep.json"), 'w', encoding="utf-8") as file:
			for item in sortedList[:keepTop]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")
		with open(os.path.join(pathToReddit, "tokenLost.json"), 'w', encoding="utf-8") as file:
			for item in sortedList[keepTop:]:
				json.dump({"word": item[0], "count": item[1]}, file)
				file.write("\n")

		pp.done()

def main():
	Proccessor().wordCountAllFiles()

if __name__ == "__main__":
	main()
