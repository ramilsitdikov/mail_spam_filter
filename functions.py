import pickle
import random
import numpy as np

SPAM_LABEL = 1
NON_SPAM_LABEL = -1

class Dataset:

    def __init__(self, data, labels):
        indexes = np.arange(len(data))
        random.shuffle(indexes)
        data = data[indexes]
        labels = labels[indexes]

        positive_set = data[labels == SPAM_LABEL]
        positive_labels = labels[labels == SPAM_LABEL]
        negative_set = data[labels == NON_SPAM_LABEL]
        negative_labels = labels[labels == NON_SPAM_LABEL]

        balanced_count = min(len(positive_set),len(negative_set))
        training_count = int(0.7*balanced_count)
        validation_count = int(0.15 * balanced_count)

        self.training_set = np.append(positive_set[:training_count],negative_set[:training_count],axis=0)
        self.training_labels = np.append(positive_labels[:training_count],negative_labels[:training_count])

        self.validation_set = np.append(positive_set[training_count:training_count+validation_count],negative_set[training_count:training_count+validation_count],axis=0)
        self.validation_labels = np.append(positive_labels[training_count:training_count+validation_count],negative_labels[training_count:training_count+validation_count])

        self.testing_set = np.append(positive_set[training_count+validation_count:balanced_count], negative_set[training_count+validation_count:balanced_count],axis=0)
        self.testing_labels = np.append(positive_labels[training_count+validation_count:balanced_count], negative_labels[training_count+validation_count:balanced_count])

POS_LABEL = 1
NEG_LABEL = -1

def measure(results,labels,weights=None):
    if not (len(results) == len(labels) and (weights is None or len(labels) == len(weights))):
        print "Error in AdaBoost::train : array of data, labels and weights dimensions is not correct!"
        return None, None, None

    if weights is None:
        weights = np.ones(len(labels))/len(labels)

    tp = np.sum(weights[np.logical_and(results == POS_LABEL, labels == POS_LABEL)])
    tn = np.sum(weights[np.logical_and(results == NEG_LABEL, labels == NEG_LABEL)])
    fp = np.sum(weights[np.logical_and(results == NEG_LABEL, labels == POS_LABEL)])
    fn = np.sum(weights[np.logical_and(results == POS_LABEL, labels == NEG_LABEL)])

    if len(labels) > 0:
        accuracy = float(tp+tn) / np.sum(weights)
    else:
        accuracy = 0.0
    if tp + fp != 0:
        precision = float(tp)/(tp+fp)
    else:
        precision = 0.0
    if tp + fn != 0:
        recall = float(tp)/(tp+fn)
    else:
        recall = 0.0
    if precision + recall != 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    return accuracy, precision, recall, fscore

def load(filepath):
    input_file = open(filepath)
    obj = pickle.load(input_file)
    input_file.close()
    return obj

def dump(obj, filepath):
    output_file = open(filepath, "wb")
    pickle.dump(obj, output_file)
    output_file.close()