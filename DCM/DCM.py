import random

from DCMFixedSize import DCMFixedSize
import numpy as np


class DCM:
    process_size = 2
    h_sizes = []
    DCMFixedSize_list = []
    rest_subsets = []
    item_num = 0
    size_distribution = []

    def __init__(self, process_size, h_sizes):
        self.process_size = process_size
        self.h_sizes = h_sizes
        self.DCMFixedSize_list = []
        for i in range(process_size):
            self.DCMFixedSize_list.append(DCMFixedSize(h_sizes[i], set_size=i+1))
        self.size_distribution = np.zeros(process_size+1)

    def train(self, data, item_num):
        self.item_num = item_num
        self.rest_subsets = []
        data_list = []
        for i in range(self.process_size):
            data_list.append([])
        for subset in data:
            if len(subset) <= self.process_size:
                data_list[len(subset) - 1].append(subset)
            else:
                self.rest_subsets.append(subset)
        for i in range(self.process_size):
            self.size_distribution[i] = len(data_list[i]) / len(data)
        self.size_distribution[-1] = len(self.rest_subsets) / len(data)
        for i in range(self.process_size):
            self.DCMFixedSize_list[i].train(data_list[i], item_num)

    def predict(self, output_size, size_distribution=None):
        predict_list = []
        # check if we use size bias plan
        if size_distribution is None:
            print("we use empirical size distribution")
            size_distribution = self.size_distribution
            print(size_distribution)
            for i in range(self.process_size):
                predict_list.extend(self.DCMFixedSize_list[i].predict(int(output_size * size_distribution[i])))
            predict_list.extend(random.choices(self.rest_subsets, k=int(output_size * size_distribution[-1])))
        else:
            print("we use size bias plan")
            print(size_distribution)
            if len(size_distribution) > self.process_size:
                print("the size distribution's size if larger than the model's process_size!")
                exit(1)
            for i in range(len(size_distribution)):
                predict_list.extend(self.DCMFixedSize_list[i].predict(int(output_size * size_distribution[i])))
        return predict_list
