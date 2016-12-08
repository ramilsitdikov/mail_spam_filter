# -*- coding: utf-8 -*-

from decision_tree import *

class AdaBoost():

    def __init__(self, number_of_rounds, max_depth, min_examples, min_entropy):
        self.alphas = []
        self.weak_classifiers = []
        self.number_of_rounds = number_of_rounds
        self.max_depth, self.min_examples, self.min_entropy = max_depth, min_examples, min_entropy
        self.errors = []

    def train(self, data, labels, weights=None):
        # Простая проверка корректности данных
        if len(data) == len(labels) and (weights is None or len(data) == len(weights)):
            training_set_count = len(data)
        else:
            print "Error in AdaBoost::train : array of data, labels and weights dimensions is not correct!"
            return False

        # Инициализация весов
        if weights is None:
            self.weights = np.ones(training_set_count) / training_set_count
        else:
            self.weights = weights

        # Обучение простых классификаторов
        for i in range(len(self.weak_classifiers), self.number_of_rounds):
            print "Training WeakClassifier", i+1, "from", self.number_of_rounds

            # Создаем простой классификатор и обучаем его
            weak_clf = WeakClassifier(self.max_depth, self.min_examples, self.min_entropy)
            weak_clf.train(data, labels, self.weights)

            # Вычисляем взвешенную ошибку классификации
            results = weak_clf.classify(data)
            errors_indicator = results != labels
            epsilon = np.sum(self.weights[errors_indicator])
            self.errors.append(epsilon)
            print "Current error:", epsilon

            # Обучить не удается из-за плохого слабого классификатора
            if np.abs(epsilon - 0.5) < 0.000001:
                print "Error in AdaBoost::train : bad WeakClassifier show result %s. This equal to 0.5!" % epsilon
                return False

            # Выбираем альфа
            alpha = np.log((1 - epsilon) / epsilon)

            # Обновляем коэффициенты
            self.weights *= np.exp(alpha * errors_indicator)

            # Нормируем полученные коэффициенты
            self.weights /= np.sum(self.weights)

            # Запоминаем классификатор с найденным коэффициентом
            self.weak_classifiers.append(weak_clf)
            self.alphas.append(alpha)

            if epsilon == 0.0:
                break

        return True

    def classify(self, data, weak_count=None):
        results = np.zeros(len(data))
        if weak_count is None:
            weak_count = len(self.weak_classifiers)
        else:
            weak_count = min(len(self.weak_classifiers),weak_count)
        for i in range(weak_count):
            results += self.alphas[i] * self.weak_classifiers[i].classify(data)
        results[results == 0] = -1
        return np.sign(results)

if __name__ == "__main__":
    number_of_rounds = 1000
    max_depth = 3
    min_examples = 1
    min_entropy = 0

    data = np.array([[0,1],
                     [0,2],
                     [1,2],
                     [1,2],
                     [1,1]])
    labels = np.array([-1,
                       -1,
                       1,
                       1,
                       -1])
    weights = np.array([1,
                        1,
                        1,
                        1,
                        1])
    wc = WeakClassifier(max_depth,min_examples,min_entropy)
    print wc.train(data,labels,weights)
    print wc.classify(data)

    ab = AdaBoost(number_of_rounds, max_depth,min_examples,min_entropy)
    print ab.train(data,labels)
    print ab.classify(data)