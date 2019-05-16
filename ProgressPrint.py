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