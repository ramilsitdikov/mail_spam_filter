# -*- coding: UTF-8 -*-
import codecs
import glob
import bag_of_words as bow
import numpy as np
from functions import Dataset, dump

test = "enron1"

path_ham = "./dataset/%s/ham/*.txt" % test
path_spam = "./dataset/%s/spam/*.txt" % test

words1 = bow.get_all_words(path_ham)
words2 = bow.get_all_words(path_spam)

words = words1
words.extend(words2)

DIC = bow.dictionary(words, 6500)

data = []
labels = []

ham_files = glob.glob(path_ham)
spam_files = glob.glob(path_spam)

for i in range(len(ham_files)):
    vector = bow.get_vectors_from_path(ham_files[i], DIC)
    data.append(vector)
    labels.append(-1)
for i in range(len(spam_files)):
    vector = bow.get_vectors_from_path(spam_files[i], DIC)
    data.append(vector)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

print("Preprocessing dataset...")
dataset = Dataset(data, labels)
print("Successful preprocessed")
print
print("Dumping dataset...")
dump(dataset, "./dataset/%s.pickle" % test)
print("Successful dump dataset")