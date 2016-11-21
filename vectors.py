# -*- coding: UTF-8 -*-
import codecs
import glob
import BoW
import numpy as np
import tensorflow as tf


test = "enron1"

path_ham = "./dataset/%s/ham/*.txt" % test
path_spam = "./dataset/%s/spam/*.txt" % test

words1 = BoW.get_all_words(path_ham)
words2 = BoW.get_all_words(path_spam)

words = words1
words.extend(words2)

DIC = BoW.dictionary(words, 6500)

data = []
labels = []

ham_files = glob.glob(path_ham)
spam_files = glob.glob(path_spam)

for i in range(len(ham_files)):
    vector = BoW.get_vectors_from_path(ham_files[i], DIC)
    data.append(vector)
    labels.append([-1])
for i in range(len(spam_files)):
    vector = BoW.get_vectors_from_path(spam_files[i], DIC)
    data.append(vector)
    labels.append([1])

data, labels = BoW.pell_mell2(data, labels)

data = np.array(data)
labels = np.array(labels)

print (data)
print (labels)



# input_data = np.array(data)
#
# output = np.array(labels)
#
# model_input = tf.placeholder(tf.float32, [None, 6500])
#
# model_output = tf.placeholder(tf.float32, [None, 1])
#
#
#
# Wo = tf.Variable(tf.random_normal([6500, 1], stddev=0.2))
#
# bo = tf.Variable(tf.random_normal([1], stddev=0.2))
#
#
#
# y = tf.sigmoid(tf.matmul(model_input, Wo) + bo)
#
# loss = tf.reduce_mean(tf.square(y - output))
#
# optimizer = tf.train.GradientDescentOptimizer(3)
#
# train = optimizer.minimize(loss)
#
#
#
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
#
# sess.run(init)
#
#
#
# print ('training...')
#
# for i in range(1000):
#
#     _, q = sess.run([train, loss], feed_dict={model_input: input_data, model_output: output})
#
#     if not i % 100 or i == 999:
#
#         print (str(i) + " " + str(q))

