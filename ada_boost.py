# -*- coding:utf-8 -*-
import numpy as np

class WeakClassifier():

    def __init__(self):
        # Параметры слабого классификатора
        pass

    def train(self, data, labels, weights):
        if len(data) == len(labels) and len(data) == len(weights):
            training_set_count = len(data)
        else:
            return False
        # Код обучения слабого классификатора
        return True

    def classify(self, data):
        results = np.ones(len(data))
        # Код классификации слабого классификатора
        return results

class AdaBoost():

    def __init__(self, number_of_rounds, weak_classifier_class):
        self.weights = None
        self.alphas = []
        self.weak_classifiers = []
        self.weak_classifier_class = weak_classifier_class
        self.number_of_rounds = number_of_rounds

    def train(self, data, labels):
        # Простая проверка корректности данных
        if len(data) == len(labels):
            training_set_count = len(data)
        else:
            return False

        # Инициализация весов
        if self.weights is None:
            self.weights = np.ones(training_set_count) / training_set_count

        # Обучение простых классификаторов
        for i in range(len(self.weak_classifiers), self.number_of_rounds):
            # Создаем простой классификатор и обучаем его
            weak_clf = self.weak_classifier_class()
            weak_clf.train(data, labels, self.weights)

            # Вычисляем взвешенную ошибку классификации
            results = weak_clf.classify(data)
            errors_indicator = results != labels
            epsilon = np.sum(self.weights[errors_indicator])

            if epsilon != 0.0:
                # Выбираем альфа
                alpha = np.log((1 - epsilon) / epsilon)

                # Обновляем коэффициенты
                self.weights *= np.exp(-alpha * errors_indicator)

                # Нормируем полученные коэффициенты
                self.weights /= np.sum(self.weights)

                # Запоминаем классификатор с найденным коэффициентом
                self.weak_classifiers.append(weak_clf)
                self.alphas.append(alpha)
            else:
                # Найденный классификатор справляется со всем в одиночку
                self.alphas = [1]
                self.weak_classifiers = [weak_clf]
                break

        return True

    def classify(self, data):
        results = np.zeros(len(data))
        for i in range(len(self.weak_classifiers)):
            results += self.alphas[i]*self.weak_classifiers[i].classify(data)
        results[results == 0] = -1
        return np.sign(results)

if False:
    data = np.array([[0,1],[0,2],[2,-1],[0,1],[2,1]])

    labels = np.array([-1,-1,1,1,-1])
    weights = np.array([0.2,0.3,0.1,0.3,0.1])
    wc = WeakClassifier()
    print wc.train(data,labels,weights)
    print wc.classify(data)

    number_of_rounds = 100
    ab = AdaBoost(number_of_rounds, WeakClassifier)
    print ab.train(data,labels)
    print ab.classify(data)