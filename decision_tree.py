from __future__ import print_function
import numpy as np
import math
from collections import Counter



class Node:
    def __init__(self, left=None,
                                right=None,
                                split_index=None,
                                split_value=None,
                                terminal=False,
                                klass=None,
                                depth=1):
        self.left = left
        self.right = right
        self.split_index = split_index
        self.split_value = split_value
        self.terminal = terminal
        self.klass = klass
        self.depth = depth


    def __str__(self):
        tabs = ("    |" * self.depth)
        if self.terminal:
            return tabs + r"Depth = {} | Class = {}".format(self.depth, self.klass)
        else:
            return tabs + r"Depth = {} | Split index = {} | Split value = {}".format(self.depth, self.split_index, self.split_value)



class DecisionTree():
    def __init__(self, max_depth=2, min_samples=0, min_entropy=0):
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_entropy = min_entropy


    def train(self, dataset, labels, weights):
        '''
        :param dataset: numpy array of lists
        :param labels: numpy array of numbers
        :param weights: numpy array of numbers
        ;return:
        '''
        split(self.root, dataset, labels, weights, self.max_depth, self.min_samples, self.min_entropy, 1)


    def classify(self, data):
        '''
        :param data: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
        :return: array of m labels -> np.array([-1, 1, 1, -1, 1])
        '''
        return np.array([predict(self.root, d) for d in data])


    def print_in_depth(self):
        print_node(self.root)



class WeakClassifier():
    def __init__(self, max_depth, min_examples, min_entropy):
        self.tree = DecisionTree(max_depth, min_examples, min_entropy)


    def train(self, data, labels, weights=None):
        if weights is None:
            weights = np.ones(labels.shape[0])
        if len(data) != len(labels) or len(data) != len(weights):
            return False
        self.tree.train(data, labels, weights)
        return True


    def classify(self, data):
        return self.tree.classify(data)



def print_node(node):
    '''
    :param node: node Node
    '''
    print(node)
    if not node.terminal:
        print_node(node.left)
        print_node(node.right)


### Make a guess
def predict(node, row):
    '''
    :param node: Node
    :param row: array of features
    :return label of klass: -1 or 1
    '''
    if node.terminal:
        return node.klass
    else:
        if row[node.split_index] <= node.split_value:
            return predict(node.left, row)
        else:
            return predict(node.right, row)


### Splitting
def split(node, dataset, labels, weights, max_depth, min_samples, min_entropy, current_depth):
    '''
    :param node: Node object
    :param dataset: numpy array of lists
    :param labels: numpy array of numbers
    :param current_depth: number
    :param max_depth: number
    :param min_samples: number
    :param min_entropy: number
    :param weights: numpy array of numbers
    :return:
    '''
    node.depth = current_depth
    if len(dataset) <= min_samples or \
                    current_depth >= max_depth or \
                    entropy(labels, weights) <= min_entropy:
        node.terminal = True
        node.klass = belong_to_class(labels)
        return

    mask, gain, split_index, split_value = get_split_mask(dataset, labels, weights)
    left_subset, right_subset = dataset[mask], dataset[~mask]
    left_labels, right_labels = labels[mask], labels[~mask]
    left_weights, right_weights = weights[mask], weights[~mask]

    if gain is None:
        node.terminal = True
        node.klass = belong_to_class(labels)
        return

    node.split_index = split_index
    node.split_value = split_value
    node.left = Node()
    node.right = Node()
    split(node.left, left_subset, left_labels, left_weights, max_depth, min_samples, min_entropy, current_depth + 1)
    split(node.right, right_subset, right_labels, right_weights, max_depth, min_samples, min_entropy, current_depth + 1)


def get_split_mask(dataset, labels, weights):
    '''
    :param dataset: numpy array of lists
    :param labels: numpy array of numbers
    :param weights: numpy array of numbers
    :return best_index: number
    :return best_split_value: number
    :return two subsets: numpy arrays of lists
    '''
    best_gain = None
    number_of_features = dataset.shape[1]
    list_of_features =np.arange(number_of_features)
    for index in list_of_features:
        unique_values = set(dataset[:, index])
        sorted_unique = sorted(unique_values)[:-1]
        for value in sorted_unique:
            mask = test_split_mask(dataset, index, value)
            gain = information_gain(labels, weights, mask)
            if gain > best_gain:
                best_gain, best_index, best_split_value, best_mask = gain, index, value, mask
    return best_mask, best_gain, best_index, best_split_value


def test_split_mask(dataset, feature_index, value_to_compare):
    '''
    :param dataset: numpy array of lists
    :param labels: numpy array of numbers
    :param feature_index: number
    :param value_to_compare: number
    :param weights: numpy array of numbers
    :return two subsets: numpy arrays of lists
    :return two labels' arrays: numpy arrays of numbers
    :return two weights' arrays: numpy arrays of numbers
    '''
    return dataset[:,feature_index] <= value_to_compare


def belong_to_class(labels):
    '''
    :param labels: numpy array of numbers
    :return: type of maximum labels' class -> -1 or 1
    '''
    counters = Counter(labels)
    return counters.most_common(1)[0][0]


### Entropy
def information_gain(labels, weights, mask):
    '''
    :param labels: numpy array of numbers
    :param left_labels: numpy array of numbers
    :param right_labels: numpy array of numbers
    :param weights: array of numbers - weights of each sample in dataset
    :param left_weights: array of numbers - weights of each sample in left_subset
    :param right_weights: array of numbers - weights of each sample in right_subset
    :return: difference between entropy befor and after split
    '''
    left_labels, right_labels = labels[mask], labels[~mask]
    left_weights, right_weights = weights[mask], weights[~mask]
    n = float(len(labels))
    n_left = len(left_labels)
    n_right = len(right_labels)
    entropy_before = entropy(labels, weights)
    entropy_left = (n_left / n) * (entropy(left_labels, left_weights) if n_left > 0 else 1)
    entropy_right = (n_right / n) * (entropy(right_labels, right_weights) if n_right > 0 else 1)

    return entropy_before - (entropy_left + entropy_right)


def entropy(labels, weights):
    '''
    :param labels: numpy array of numbers
    :param weights: numpy array of numbers
    :return: entropy of dataset
    '''
    n = float(len(labels))
    mask = labels == -1
    n_class1 = sum(weights[mask])
    n_class2 = sum(weights[~mask])
    class1 = (n_class1 / n) * math.log(n_class1 / n if n_class1 > 0 else 1, 2)
    class2 = (n_class2 / n) * math.log(n_class2 / n if n_class2 > 0 else 1, 2)
    entropy = class1 + class2

    return -entropy



if __name__ == "__main__":
    import time
    X = np.array([[5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5.0,3.6,1.4,0.2], [5.4,3.9,1.7,0.4], [4.6,3.4,1.4,0.3], [5.0,3.4,1.5,0.2], [4.4,2.9,1.4,0.2], [4.9,3.1,1.5,0.1], [5.4,3.7,1.5,0.2], [4.8,3.4,1.6,0.2], [4.8,3.0,1.4,0.1], [4.3,3.0,1.1,0.1], [5.8,4.0,1.2,0.2], [5.7,4.4,1.5,0.4], [5.4,3.9,1.3,0.4], [5.1,3.5,1.4,0.3], [5.7,3.8,1.7,0.3], [5.1,3.8,1.5,0.3], [5.4,3.4,1.7,0.2], [5.1,3.7,1.5,0.4], [4.6,3.6,1.0,0.2], [5.1,3.3,1.7,0.5], [4.8,3.4,1.9,0.2], [5.0,3.0,1.6,0.2], [5.0,3.4,1.6,0.4], [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2], [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1], [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1], [5.0,3.2,1.2,0.2], [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2], [5.1,3.4,1.5,0.2], [5.0,3.5,1.3,0.3], [4.5,2.3,1.3,0.3], [4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4], [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2], [4.6,3.2,1.4,0.2], [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3], [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6], [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3], [5.2,2.7,3.9,1.4], [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0], [6.1,2.9,4.7,1.4], [5.6,2.9,3.6,1.3], [6.7,3.1,4.4,1.4], [5.6,3.0,4.5,1.5], [5.8,2.7,4.1,1.0], [6.2,2.2,4.5,1.5], [5.6,2.5,3.9,1.1], [5.9,3.2,4.8,1.8], [6.1,2.8,4.0,1.3], [6.3,2.5,4.9,1.5], [6.1,2.8,4.7,1.2], [6.4,2.9,4.3,1.3], [6.6,3.0,4.4,1.4], [6.8,2.8,4.8,1.4], [6.7,3.0,5.0,1.7], [6.0,2.9,4.5,1.5], [5.7,2.6,3.5,1.0], [5.5,2.4,3.8,1.1], [5.5,2.4,3.7,1.0], [5.8,2.7,3.9,1.2], [6.0,2.7,5.1,1.6], [5.4,3.0,4.5,1.5], [6.0,3.4,4.5,1.6], [6.7,3.1,4.7,1.5], [6.3,2.3,4.4,1.3], [5.6,3.0,4.1,1.3], [5.5,2.5,4.0,1.3], [5.5,2.6,4.4,1.2], [6.1,3.0,4.6,1.4], [5.8,2.6,4.0,1.2], [5.0,2.3,3.3,1.0], [5.6,2.7,4.2,1.3], [5.7,3.0,4.2,1.2], [5.7,2.9,4.2,1.3], [6.2,2.9,4.3,1.3], [5.1,2.5,3.0,1.1], [5.7,2.8,4.1,1.3]])
    y = np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    # X = np.array([[1, 0],
    #               [0, 1],
    #               [1, 1],
    #               [0, 0]])
    # y = np.array([1, -1, -1, 1])

    start = time.time()
    tree = DecisionTree()
    tree.train(X, y, np.ones(y.shape[0]))
    predicted_y = tree.classify(X)
    end = time.time()


    difference = np.sum(np.abs(predicted_y - y))
    print("Number of different elements =", difference/2)
    tree.print_in_depth()
    print("Elapsed time = ", end - start)