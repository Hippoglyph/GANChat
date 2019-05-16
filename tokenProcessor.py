import re
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os

pathToReddit = os.path.realpath("Reddit")

class tokenProcessor():
	def __init__(self):
		self.replaceTokens = {"url": "xx-url-xx", "user": "xx-user-xx", "sub": "xx-subreddit-xx", "number": " xx-number-xx ", "hashtag": "xx-hashtag-xx", "unknown": "xx-unknown-xx", "end": "xx-end-xx", "start": "xx-start-xx", "pad": "xx-pad-xx"}
		#self.splitString = r"([\s\"\(\)\[\]\{\}\,\.\?\!\%\&\:\;\-\=\\\/\*\^]" + "".join(["|"+self.replaceTokens[key] for key in self.replaceTokens])+")"
		#self.splitString = r"([^\w']" + "".join(["|"+self.replaceTokens[key] for key in self.replaceTokens])+")"

		self.wordToIndexFileName = "tokenToIndex.json"

		self.tokenToInt = None
		self.intToToken = None

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
		#text = re.sub(r"[\r\n]+", "\n", text) #Newlines
		text = re.sub(r"(?<=\W)\/?u\/[^\s)\]\}]+", self.replaceTokens["user"], " " +text)[1:] #User
		text = re.sub(r"(?<=\W)\/?r\/[^\s)\]\}]+", self.replaceTokens["sub"], " " +text)[1:] #Subreddit
		text = re.sub(r"#\w+", self.replaceTokens["hashtag"], text) #Hashtag


		text = re.sub(r"(?<=\s)lpts?\s?\:?\-?\??\!?('s)?;?", "", " " +text)[1:] #lpt
		text = re.sub(r"(?<=\s)eli5s?\s?\:?\-?\??\!?('s)?;?", "", " " +text)[1:] #eli5

		text = re.sub(r"\d+([\.,]\d+)*", self.replaceTokens["number"], text) #numbers
		#text = re.sub(r"\d+", self.replaceTokens["number"], text) #numbers

		text = re.sub(r"\.{3,}", "...", text) #dots
		text = re.sub(r"(?<!\.)\.{2}(?!\.)", ".", text) #dots

		text = re.sub(r"(´´|''|``)", "\"", text) #cite
		
		text = re.sub(r"xx_url_xx", self.replaceTokens["url"], text) #DataCreator
		text = re.sub(r"xx_user_xx", self.replaceTokens["user"], text) #DataCreator
		text = re.sub(r"xx_subreddit_xx", self.replaceTokens["sub"], text) #DataCreator
		
		text = re.sub(r"\s+", " ", text)	#Whitespace
		return text

	def tokenize(self, text):
		#return list(filter(None,re.split(self.splitString,text)))
		return nltk.word_tokenize(text)

	def detokenize(self, tokens):
		text = TreebankWordDetokenizer().detokenize(tokens)
		text = re.sub(r'\s+\.\s+', '. ', text)
		text = re.sub(r"\"", "\" ", text) #cite
		text = re.sub(r"``", " \"", text) #cite
		#text = re.sub(r'\s*,\s*', ', ', text)
		#text = re.sub(r'\s*\?\s*', '? ', text)
		return text

	def initTokenMaps(self):
		print("Initilizing token maps")
		if not os.path.exists(os.path.join(pathToReddit, self.wordToIndexFileName)):
			print("No token index folder")
			return

		self.tokenToInt = {}
		self.intToToken = {}

		with open(os.path.join(pathToReddit, self.wordToIndexFileName), "r") as tokenToIndexFile:
			for token in tokenToIndexFile.readlines():
				self.tokenToInt[token["word"]] = token["id"]
				self.intToToken[token["id"]] = token["word"]
