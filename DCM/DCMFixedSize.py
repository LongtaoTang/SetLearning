import numpy as np
import random


class DCMFixedSize:
    set_size = 0
    H_size = 0
    H_list = list([])
    H_p = np.zeros(H_size)
    item_num = 0
    item_p = np.zeros(item_num)
    item_cum_p = np.zeros(item_num)

    def __init__(self, h_size, set_size):
        self.set_size = set_size
        self.H_size = int(h_size)
        self.H_list = list([])
        self.H_p = np.zeros(self.H_size)
        self.item_num = 0
        self.item_p = np.zeros(self.item_num)

    def train(self, data, item_num):
        self.H_list = list([])
        self.item_num = item_num
        self.item_p = np.zeros(item_num)
        self.item_cum_p = np.zeros(item_num)

        subset_dict = dict()
        for i in range(len(data)):
            subset = data[i]
            if len(subset) != self.set_size:
                print("the data's subset size is not equal to the model's set_size!")
                exit(1)
            subset = list(subset)
            subset.sort()
            subset = tuple(subset)
            if subset in subset_dict:
                subset_dict[subset] += 1
            else:
                subset_dict[subset] = 1
        subset_list = list()
        for subset in subset_dict:
            subset_list.append([subset_dict[subset], subset])

        subset_list.sort(key=lambda elem: elem[0], reverse=True)

        for i in range(len(subset_list)):
            if i >= self.H_size:
                subset = subset_list[i][1]
                for x in subset:
                    self.item_p[x] += subset_list[i][0]
            else:
                self.H_p[i] = subset_list[i][0] / len(data)
                self.H_list.append(set(subset_list[i][1]))
        sum_p = 0
        for i in range(item_num):
            sum_p += self.item_p[i]
            self.item_cum_p[i] += sum_p
        for i in range(item_num):
            self.item_p[i] = self.item_p[i] / sum_p
            self.item_cum_p[i] = self.item_cum_p[i] / sum_p

    def predict(self, output_size):
        print("output_size: ", output_size, "set_size: ", self.set_size)
        output_list = list()
        for subset_index in range(output_size):
            sum_p = 0
            r = random.random()
            for i in range(self.H_size):
                sum_p += self.H_p[i]
                if sum_p > r:
                    output_list.append(self.H_list[i])
                    break
                if i == self.H_size - 1:
                    # we choose form other part
                    while True:
                        subset = random.choices(range(self.item_num), cum_weights=self.item_cum_p, k=self.set_size)
                        subset = set(subset)
                        if subset in self.H_list:
                            continue
                        if len(subset) != self.set_size:
                            continue
                        break
                    output_list.append(subset)
        return output_list
