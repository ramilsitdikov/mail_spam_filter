import numpy as np
import math


class Node:
    def __init__(self, left=None,
                                right=None,
                                split_index=None,
                                split_value=None,
                                terminal=False,
                                klass=None):
        self.left = left
        self.right = right
        self.split_index = split_index
        self.split_value = split_value
        self.terminal = terminal
        self.klass = klass


class DecisionTree():
    def __init__(self, max_depth=2):
        self.root = Node()
        self.max_depth = max_depth


    def train(self, X, y):
        '''
        :param X: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
        :param Y: array m labels -> np.array([-1, 1, -1, 1, 1])
        '''
        dataset = make_dataset(X, y)
        current_depth = 0
        split(self.root, dataset, self.max_depth, current_depth + 1)


    def classify(self, data):
        '''
        :param data: matrix m samples, n features each -> np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])
        :return: array of m labels -> np.array([-1, 1, 1, -1, 1])
        '''
        results = list()
        for d in data:
            result.append(predict(self.root, d))
        return np.array(results)



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
    if row[node.index] < node.split_value:
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
def split(node, dataset, max_depth, current_depth):
    '''
    :param node: Node
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param current_depth: current depth of tree -> number
    :return:
    '''
    split_index, split_value, left_subset, right_subset = get_split(dataset)
    if not left_subset or not right_subset:
        node.terminal = True
        return
    if depth >= max_depth:
        node.split_index = split_index
        node.split_value = split_value
        node.left = Node(terminal = True, klass = belong_to_klass(left_subset))
        node.right = Node(terminal = True, klass = belong_to_klass(right_subset))
        return
    node.left = Node()
    node.right = Node()
    split(node.left, left_subset, current_depth + 1)
    split(node.right, right_subset, current_depth + 1)


def test_split(dataset, feature_index, value_to_compare):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    :param feature_index: number of feature to split
    :param value_to_compare: value to compare given feature
    :return: two subsets like np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    '''
    left_subset, right_subset = list(), list()
    for row in dataset:
        if row[feature_index] < value_to_compare:
            left_subset.append(row)
        else:
            right_subset.append(row)

    return left_subset, right_subset


def get_split(dataset):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    :return: best_index -> number
    :return: best_split_value -> number
    :return: two subsets like np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    '''
    number_of_features = len(dataset[0].features)
    class_values = [-1, 1]
    best_gain, best_index, best_split_value = 0
    best_left, best_right = None, None
    for index in xrange(number_of_features):
        for row in dataset:
            left_subset, right_subset = test_split(index, row[index], dataset)
            current_gain = information_gain(dataset, left_subset, right_subset)
            if current_gain > best_gain:
                best_gain, best_index, best_split_value = current_gain, index, row[index]
                best_left, best_right = left_subset, right_subset

    return best_index, best_split_value, best_left, best_right


def belong_to_klass(subset):
    '''
    :param subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "klass": Y[i] }, ...])
    :return: type of maximum labels' klass -> -1 or 1
    '''
    outcomes = [row["klass"] for row in subset]
    return max(set(outcomes), key=outcomes.count)



### Entropy
def entropy(dataset):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :return: entropy of dataset
    '''
    n = len(dataset)
    n_class1 = 0
    n_class2 = 0
    for element in dataset:
        if element.klass == -1:
            n_class1 += 1
        elif element.klass == 1:
            n_class2 += 1
        else:
            raise Exception('Error in dataset: class label not from [-1, 1].')
    class1 = (n_class1 / n) * math.log(n_class1 / n)
    class2 = (n_class2 / n) * math.log(n_class2 / n)
    entropy = class1 + class2
    return -entropy


def information_gain(dataset, left_subset, right_subset):
    '''
    :param dataset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param left_subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :param right_subset: array of dictionaries like { "features": list_of_features, "class": label } -> np.array([..., { "features": X[i], "class": Y[i] }, ...])
    :return: difference between entropy befor and after split
    '''
    n = dataset.shape[0]
    n_left = left_subset.shape[0]
    n_right = right_subset.shape[0]
    entropy_before = entropy(dataset)
    weighted_entropy_left = (n_left / n) * entropy(left_subset)
    weighted_entropy_right = (n_right / n) * entropy(right_subset)
    return entropy_before - (weighted_entropy_left + weighted_entropy_right)