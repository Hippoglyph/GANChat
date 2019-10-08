import os
import matplotlib.pyplot as plt
import numpy as np

seq2seqSave = os.path.join(os.path.dirname(__file__), "Seq2SeqGAN", "save")
seqSave = os.path.join(os.path.dirname(__file__), "SeqGAN", "save")
experimentName = "real-data-target-and-pretrain-200-more-data.txt"

file = os.path.join(seq2seqSave, experimentName)
adversarialStart = 120
plotName = "SeqGAN"
title = "SeqGAN Copy"
xlabel = "Epochs"
ylabel = "NLL scores"

replaceYticks = False
adYtick = False

epochs = []
scores = []
with open(file, "r") as f:
	for row in f.readlines():
		epoch, score = row.split()
		epochs.append(int(epoch))
		scores.append(float(score))


fig, ax = plt.subplots()
ax.plot(epochs, scores, label=plotName)
plt.vlines(x=adversarialStart, linestyle="--", color="b", label="Adversarial training start", ymin=min(scores)//0.1*0.1, ymax=max(scores)//0.1*0.1)
plt.hlines(y=min(scores), color="r", linestyle="--", xmin=min(epochs), xmax=max(epochs))
ax.grid()
plt.autoscale(enable=True, axis='x', tight=True)

if replaceYticks:
	record = float("inf")
	replace = 0
	for i in range(len(list(plt.yticks()[0]))):
		dist = abs(min(scores) - list(plt.yticks()[0])[i])
		if dist < record:
			record = dist
			replace = i

	yticks = []
	for i in range(len(list(plt.yticks()[0]))):
		if i != replace:
			yticks.append(list(plt.yticks()[0])[i])
	yticks.append(min(scores))
	plt.yticks(yticks)
elif adYtick:
	plt.yticks(list(plt.yticks()[0]) + [min(scores)])

plt.xticks(list(plt.xticks()[0]) + [adversarialStart])
ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
plt.legend()
plt.show()