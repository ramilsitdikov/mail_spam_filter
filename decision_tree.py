from __future__ import print_function
import numpy as np
import math


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


    def train(self, X, y, weights):
        '''
        :param X: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
        :param Y: array m labels -> np.array([-1, 1, -1, 1, 1])
        :param weights: array of numbers - weights of each sample in dataset
        ;return:
        '''
        dataset = make_dataset(X, y)
        current_depth = 0
        split(self.root, dataset, self.max_depth, self.min_samples, self.min_entropy, current_depth + 1, weights)


    def classify(self, data):
        '''
        :param data: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
        :return: array of m labels -> np.array([-1, 1, 1, -1, 1])
        '''
        results = list()
        for d in data:
            results.append(predict(self.root, d))
        return np.array(results)

    def print_in_depth(self):
        print_node(self.root)



def print_node(node):
    '''
    :param node: node Node
    '''
    print(node)
    if not node.terminal:
        print_node(node.left)
        print_node(node.right)


def make_dataset(X, Y):
    '''
    :param X: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
    :param Y: array m labels -> np.array([-1, 1, -1, 1, 1])
    :return: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    '''
    if not X.shape[0] == Y.size:
        raise Exception('Error in feature vectors or lists: lengths not equal.')
    dataset = []
    for i in xrange(X.shape[0]):
        dataset.append({ "features": X[i], "class": Y[i] })
    return np.array(dataset)


### Make a guess
def predict(node, row):
    '''
    :param node: Node
    :param row: array of features -> np.array([-1, 1, -1, 1, 1])
    :return: label of klass -> -1 or 1
    '''
    if row[node.split_index] < node.split_value:
        if node.left.terminal:
            return node.left.klass
        else:
            return predict(node.left, row)
    else:
        if node.right.terminal:
            return node.right.klass
        else:
            return predict(node.right, row)


### Splitting
def split(node, dataset, max_depth, min_samples, min_entropy, current_depth, weights):
    '''
    :param node: Node
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param current_depth: current depth of tree -> number
    :param weights: array of numbers - weights of each sample in dataset
    :return:
    '''
    gain, split_index, split_value, left_subset, right_subset, left_weights, right_weights = get_split(dataset, weights)
    if len(left_subset) <= min_samples or \
       len(right_subset) <= min_samples or \
       current_depth >= max_depth or \
       gain <= min_entropy:
           node.terminal = True
           node.klass = belong_to_class(dataset)
           node.depth = current_depth
           return
    node.split_index = split_index
    node.split_value = split_value
    node.left = Node()
    node.right = Node()
    node.depth = current_depth
    split(node.left, left_subset, max_depth, min_samples, min_entropy, current_depth + 1, left_weights)
    split(node.right, right_subset, max_depth, min_samples, min_entropy, current_depth + 1, right_weights)


def belong_to_class(subset):
    '''
    :param subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    :return: type of maximum labels' klass -> -1 or 1
    '''
    outcomes = [row["class"] for row in subset]
    return max(set(outcomes), key=outcomes.count)


def get_split(dataset, weights):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    :return: best_index -> number
    :return: best_split_value -> number
    :return: two subsets like np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    '''
    number_of_features = len(dataset[0]["features"])
    class_values = [-1, 1]
    best_gain, best_index, best_split_value = 0, 0, 0
    best_left, best_right = [], []
    best_left_weights, best_right_weights = [], []
    for index in xrange(number_of_features):
        for row in dataset:
            left_subset, right_subset, left_weights, right_weights = test_split(dataset, index, row["features"][index], weights)
            current_gain = information_gain(dataset, left_subset, right_subset, weights, left_weights, right_weights)
            if current_gain > best_gain:
                best_gain, best_index, best_split_value = current_gain, index, row["features"][index]
                best_left, best_right = left_subset, right_subset
                best_left_weights, best_right_weights = left_weights, right_weights

    return best_gain, best_index, best_split_value, best_left, best_right, best_left_weights, best_right_weights


def test_split(dataset, feature_index, value_to_compare, weights):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param feature_index: number of feature to split
    :param value_to_compare: value to compare given feature
    :param weights: array of numbers - weights of each sample in dataset
    :return: two subsets like np.array([..., { "features": X[i], "class": Y[i] }, ...]), two arrays of weights
    '''
    left_subset, right_subset = list(), list()
    left_weights, right_weights = list(), list()
    for row in xrange(dataset.shape[0]):
        if dataset[row]["features"][feature_index] < value_to_compare:
            left_subset.append(dataset[row])
            left_weights.append(weights[row])
        else:
            right_subset.append(dataset[row])
            right_weights.append(weights[row])

    return np.array(left_subset), np.array(right_subset), left_weights, right_weights


def information_gain(dataset, left_subset, right_subset, dataset_weights, left_weights, right_weights):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param left_subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param right_subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param dataset_weights: array of numbers - weights of each sample in dataset
    :param left_weights: array of numbers - weights of each sample in left_subset
    :param right_weights: array of numbers - weights of each sample in right_subset
    :return: difference between entropy befor and after split
    '''
    n = dataset.shape[0]
    n_left = left_subset.shape[0]
    n_right = right_subset.shape[0]
    entropy_before = entropy(dataset, dataset_weights)
    weighted_entropy_left = (n_left / n) * (entropy(left_subset, left_weights) if len(left_subset) > 0 else 1)
    weighted_entropy_right = (n_right / n) * (entropy(right_subset, right_weights) if len(right_subset) > 0 else 1)

    return entropy_before - (weighted_entropy_left + weighted_entropy_right)


### Entropy
def entropy(dataset, weights):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param weights: array of numbers - weights of each sample in dataset
    :return: entropy of dataset
    '''
    n = dataset.shape[0]
    n_class1 = 0
    n_class2 = 0
    for i in xrange(n):
        if dataset[i]["class"] == -1:
            n_class1 += weights[i]
        elif dataset[i]["class"] == 1:
            n_class2 += weights[i]
        else:
            raise Exception('Error in dataset: class label not from [-1, 1].')
    class1 = (n_class1 / n) * math.log(n_class1 / n if n_class1 > 0 else 1)
    class2 = (n_class2 / n) * math.log(n_class2 / n if n_class2 > 0 else 1)
    entropy = class1 + class2

    return -entropy



if __name__ == "__main__":
    X = np.array([[5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5.0,3.6,1.4,0.2], [5.4,3.9,1.7,0.4], [4.6,3.4,1.4,0.3], [5.0,3.4,1.5,0.2], [4.4,2.9,1.4,0.2], [4.9,3.1,1.5,0.1], [5.4,3.7,1.5,0.2], [4.8,3.4,1.6,0.2], [4.8,3.0,1.4,0.1], [4.3,3.0,1.1,0.1], [5.8,4.0,1.2,0.2], [5.7,4.4,1.5,0.4], [5.4,3.9,1.3,0.4], [5.1,3.5,1.4,0.3], [5.7,3.8,1.7,0.3], [5.1,3.8,1.5,0.3], [5.4,3.4,1.7,0.2], [5.1,3.7,1.5,0.4], [4.6,3.6,1.0,0.2], [5.1,3.3,1.7,0.5], [4.8,3.4,1.9,0.2], [5.0,3.0,1.6,0.2], [5.0,3.4,1.6,0.4], [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2], [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1], [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1], [5.0,3.2,1.2,0.2], [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2], [5.1,3.4,1.5,0.2], [5.0,3.5,1.3,0.3], [4.5,2.3,1.3,0.3], [4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4], [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2], [4.6,3.2,1.4,0.2], [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3], [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6], [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3], [5.2,2.7,3.9,1.4], [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0], [6.1,2.9,4.7,1.4], [5.6,2.9,3.6,1.3], [6.7,3.1,4.4,1.4], [5.6,3.0,4.5,1.5], [5.8,2.7,4.1,1.0], [6.2,2.2,4.5,1.5], [5.6,2.5,3.9,1.1], [5.9,3.2,4.8,1.8], [6.1,2.8,4.0,1.3], [6.3,2.5,4.9,1.5], [6.1,2.8,4.7,1.2], [6.4,2.9,4.3,1.3], [6.6,3.0,4.4,1.4], [6.8,2.8,4.8,1.4], [6.7,3.0,5.0,1.7], [6.0,2.9,4.5,1.5], [5.7,2.6,3.5,1.0], [5.5,2.4,3.8,1.1], [5.5,2.4,3.7,1.0], [5.8,2.7,3.9,1.2], [6.0,2.7,5.1,1.6], [5.4,3.0,4.5,1.5], [6.0,3.4,4.5,1.6], [6.7,3.1,4.7,1.5], [6.3,2.3,4.4,1.3], [5.6,3.0,4.1,1.3], [5.5,2.5,4.0,1.3], [5.5,2.6,4.4,1.2], [6.1,3.0,4.6,1.4], [5.8,2.6,4.0,1.2], [5.0,2.3,3.3,1.0], [5.6,2.7,4.2,1.3], [5.7,3.0,4.2,1.2], [5.7,2.9,4.2,1.3], [6.2,2.9,4.3,1.3], [5.1,2.5,3.0,1.1], [5.7,2.8,4.1,1.3]])
    y = np.array([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    tree = DecisionTree(max_depth = 5)

    tree.train(X, y, np.ones(y.shape[0]))
    predicted_y = tree.classify(X)
    difference = np.sum(np.abs(predicted_y - y))
    print("Number of different elements =", difference/2)
    tree.print_in_depth()
