import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
import tflearn
import json

## Fetch the training data
with open("intents.json") as file:
	data = json.load(file)

##Tokenize

words = []
docs = []
labels = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs.append(pattern)

	if intent["tag"] not in labels:
		labels.append(intent["tag"])


