# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:55:26 2021

@author: illusory
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#import black

class LinearSVM(object):
    def __init__(self, dataset_size, vector_size,epsilon):
        self.__multipliers = np.zeros(dataset_size, np.float_)
        self.weight_vec = np.zeros(vector_size, np.float_)
        self.bias = 0
        self.epsilon = epsilon
    def train(self, dataset, iteration_num):
        dataset = np.array(dataset)
        for k in tqdm(range(iteration_num)):
            self.__update(dataset, k)

    def __update(self, dataset, k):
        for i in range(dataset.__len__() // 2):
            j = (dataset.__len__() // 2 + i + k) % dataset.__len__()
            record_i = dataset[i]
            record_j = dataset[j]
            self.__sequential_minimal_optimization(dataset, record_i, record_j, i, j)
            self.__update_weight_vec(dataset)
            self.__update_bias(dataset)
  
    def __sequential_minimal_optimization(self, dataset, record_i, record_j, i, j):
        label_i = record_i[-1]
        vector_i = np.array(record_i[0])
        label_j = record_j[-1]
        vector_j = np.array(record_j[0]) 
        error_i = np.dot(self.weight_vec, vector_i) + self.bias - label_i
        if abs(error_i)<self.epsilon:
            error_i = 0
        error_j = np.dot(self.weight_vec, vector_j) + self.bias - label_j
        if abs(error_j)<self.epsilon:
            error_i = 0
        eta = np.dot(vector_i - vector_j, vector_i - vector_j)
        unclipped_i = self.__multipliers[i] + label_i * (error_j - error_i) / eta
        constant = -self.__calculate_constant(dataset, i, j)
        multiplier = self.__quadratic_programming(unclipped_i, label_i, label_j, i, j)
        if multiplier >= 0:
            self.__multipliers[i] = multiplier
            self.__multipliers[j] = (constant - multiplier * label_i) * label_j
 
    def __update_bias(self, dataset):
        sum_bias = 0
        count = 0
        for k in range(self.__multipliers.__len__()):
            if self.__multipliers[k] != 0:
                label = dataset[k][-1]
                vector = np.array(dataset[k][0])
                sum_bias += 1 / label - np.dot(self.weight_vec, vector)
                count += 1
        if count == 0:
            self.bias = 0
        else:
            self.bias = sum_bias / count

    def __update_weight_vec(self, dataset):
        weight_vector = np.zeros(dataset[0][0].__len__())
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            vector = np.array(dataset[k][0])
            weight_vector += self.__multipliers[k] * label * vector
        self.weight_vec = weight_vector
 
    def __calculate_constant(self, dataset, i, j):
        label_i = dataset[i][-1]
        label_j = dataset[j][-1]
        dataset[i][-1] = 0
        dataset[j][-1] = 0
        sum_constant = 0
        for k in range(dataset.__len__()):
            label = dataset[k][-1]
            sum_constant += self.__multipliers[k] * label
        dataset[i][-1] = label_i
        dataset[j][-1] = label_j
        return sum_constant
 
    def __quadratic_programming(self, unclipped_i, label_i, label_j, i, j):
        multiplier = -1
        if label_i * label_j == 1:
            boundary = self.__multipliers[i] + self.__multipliers[j]
            if boundary >= 0:
                if unclipped_i <= 0:
                    multiplier = 0
                elif unclipped_i < boundary:
                    multiplier = unclipped_i
                else:
                    multiplier = boundary
        else:
            boundary = max(0, self.__multipliers[i] - self.__multipliers[j])
            if unclipped_i <= boundary:
                multiplier = boundary
            else:
                multiplier = unclipped_i
        return multiplier
 
    def predict(self, vector):
        result = np.dot(self.weight_vec, np.array(vector)) + self.bias
        if result >= 0:
            return 1
        else:
            return -1
 
    def __str__(self):
        return "multipliers:" + self.__multipliers.__str__() + '\n' + \
                "weight_vector:" + self.weight_vec.__str__() + '\n' + \
               "bias:" + self.bias.__str__()

class SVM():
    def __init__(self):
        self.dataset = {"data":[],"label":[]}

    def linerable(self,datasize=500):
        self.dataset["label"] = np.zeros((2 * datasize, 1))

        x1 = np.reshape(np.random.normal(1, 0.3, datasize), (datasize, 1))
        y1 = np.reshape(np.random.normal(1, 0.8, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((x1, y1), axis=1)
        self.dataset["label"][0:datasize, :] = -1

        x2 = np.reshape(np.random.normal(3, 0.4, datasize), (datasize, 1))
        y2 = np.reshape(np.random.normal(3, 0.5, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((self.dataset["data"], np.concatenate((x2, y2), axis=1)), axis=0)
        self.dataset["label"][datasize:2 * datasize, :] = 1

    def unlinerable(self,datasize=500):
        self.dataset["label"] = np.zeros((2 * datasize, 1))

        x1 = np.reshape(np.random.normal(1, 0.9, datasize), (datasize, 1))
        y1 = np.reshape(np.random.normal(1, 0.5, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((x1, y1), axis=1)
        self.dataset["label"][0:datasize, :] = 0

        x2 = np.reshape(np.random.normal(2, 0.8, datasize), (datasize, 1))
        y2 = np.reshape(np.random.normal(2, 0.8, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((self.dataset["data"], np.concatenate((x2, y2), axis=1)), axis=0)
        self.dataset["label"][datasize:2 * datasize, :] = 1

a = SVM()
a.linerable()
dataset = [[list(a.dataset["data"][i]),int(a.dataset["label"][i][0])] for i in range(len(a.dataset["data"]))]

linearSVM = LinearSVM(dataset.__len__(), dataset[0][0].__len__(),0.1)
linearSVM.train(dataset, 50)

correct = 0
for record in dataset:
    vector = record[0]
    label = record[-1]
    if label == 1:
        plt.plot(vector[0], vector[1], 'r-o')
    else:
        plt.plot(vector[0], vector[1], 'g-o')

    predict = linearSVM.predict(vector)
    if predict == label:
        correct += 1

print("acc:",correct/len(dataset))
x1 = np.linspace(0, 6, 100)
x2 = (-linearSVM.bias - linearSVM.weight_vec[0] * x1) / linearSVM.weight_vec[1]
plt.plot(x1, x2)
plt.show()