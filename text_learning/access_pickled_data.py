#!/usr/bin/python

import os
import pickle
import re
import sys

print "loading processed emails"
word_data = pickle.load(open("your_word_data.pkl", "r") )
from_data = pickle.load(open("your_email_authors.pkl", "r") )

print("word_data[152] is: %s" % word_data[152])

### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit_transform(word_data)

print("vocabulary size is: %d" % len(vectorizer.vocabulary_))
for k, v in vectorizer.vocabulary_.iteritems():
    if v == 34597:
        print("word number %d is: %s" % (v, k))
