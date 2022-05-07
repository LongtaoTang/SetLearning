import numpy as np
from scipy import optimize as op
import random
import time
import pickle
from randomwalk_fast_neighborhood_v3 import *
from multiprocessing import Pool
from utility.util import *


class Order:
    def __init__(self, frequency, cardinality, itemset):
        self.frequency = frequency
        self.cardinality = cardinality
        self.itemset = itemset


# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"

initial_learning_rate = 0.01
factor = 0.999
steps = 3000
batch_size = 128
d = 10  # length of feature vector
cardinality_constraint = 10
# -----------------------------------------------------------------------------------------------------


def iter(batch_size, orderlist, param, neighborhood):
    GradX = np.zeros(param['X'].shape)
    Gradx_t = np.zeros(param['x_t'].shape)
    likelihood = np.array(0.)

    # cp_param = {'X': cp.array(param['X']), 'x_t': cp.array(param['x_t']), 'n': param['n']}

    count = 0
    for i in range(batch_size):
        id = random.randint(0, len(orderlist) - 1)
        r = cal_prob_grad(orderlist[id].itemset, param, neighborhood)
        if r['hit'] > 1e-40:
            GradX += r['GradX'] / r['hit']
            Gradx_t += r['Gradx_t'] / r['hit']
            likelihood += np.log(r['hit'])
        else:
            count += 1
    print('count: ', count)

    # mean = (np.sum(GradX) + np.sum(Gradx_t)) / (GradX.shape[0] * GradX.shape[1] + Gradx_t.shape[0])
    # std = np.sqrt((np.sum((GradX - mean) * (GradX - mean)) + np.sum((Gradx_t - mean) *
    #                (Gradx_t - mean))) / (GradX.shape[0] * GradX.shape[1] + Gradx_t.shape[0]))
    std = 1
    GradX = (GradX) / std
    print(np.max(GradX), np.min(GradX), Gradx_t, likelihood)
    Gradx_t /= std
    likelihood /= batch_size - count
    return {'GradX': GradX, 'Gradx_t': Gradx_t,
            'likelihood': likelihood}


def train(orderlist, param, neighborhood):
    X = param['X']
    x_t = param['x_t']

    learning_rate = initial_learning_rate
    l = []
    for step in range(steps):
        t1 = time.time()
        ret = iter(batch_size, orderlist, param, neighborhood)
        X += learning_rate * ret['GradX']
        # X = X / (np.std(X) * 100)
        x_t += learning_rate * ret['Gradx_t']
        learning_rate *= factor
        t2 = time.time()
        print(ret['likelihood'])
        l.append(ret['likelihood'])
        print('elapsed time: ', t2 - t1)

    return l


def run(parameters):
    data = pickle.load(open(parameters['data_path'], 'rb'))

    item_num = parameters['item_num']

    # distr = {1: 0, 2: 0, 3: 0}
    distr = {x: 0 for x in range(1, cardinality_constraint + 1)}

    training_data = []
    for subset in data:
        cardinality = len(subset)
        if cardinality > cardinality_constraint:
            continue
        # if cardinality < lb_card:
        #     continue
        distr[cardinality] += 1
        training_data.append(Order(1, cardinality, subset))

    neighborhood = np.ones((item_num + 1, item_num + 1))
    for i in range(item_num + 1):
        neighborhood[i, i] = 0
        neighborhood[i, 0] = 0

    print('number of orders: ', len(training_data))
    X = np.random.normal(size=(item_num + 1, d), scale=0.01)
    x_t = np.zeros((item_num + 1,))
    x_t[:] = 7
    # x_t = np.array(-1.5)
    # print(x_t.shape)
    param = {'X': X, 'x_t': x_t, 'n': item_num}
    ret = train(training_data, param, neighborhood)

    # Store training result
    trans_matrix = get_trans([x for x in range(item_num + 2)], param, neighborhood, 'np')
    pickle.dump(X, open(parameters['embedding_path'], 'wb'))
    pickle.dump(trans_matrix, open(parameters['trans_matrix_path'], 'wb'))


if __name__ == '__main__':
    info = pickle.load(open("../tasks/task" + task_index + "/info.pickle", "rb"))

    time_str = timestamp()
    time_str = time_str.replace(':', '_')
    mkdir("../tasks/task" + task_index + "/RW")
    mkdir("../tasks/task" + task_index + "/RW/" + time_str)

    root_path = "../tasks/task" + task_index + "/RW/" + time_str
    log_path = root_path + "/log_train.txt"

    log_file = open(log_path, 'w')
    log_file.write("we use RW method for training." + "\n")
    log_file.write("initial_learning_rate: " + str(initial_learning_rate) + "\n")
    log_file.write("factor: " + str(factor) + "\n")
    log_file.write("batch_size: " + str(batch_size) + "\n")
    log_file.write("embedding dim: " + str(d) + "\n")
    log_file.write("max subset size we use: " + str(cardinality_constraint) + "\n")
    log_file.write("how many steps we train: " + str(steps) + "\n")

    pool = Pool(8)
    results = []
    for group_index in range(int(info['#group'])):
        parameters = dict({'data_path': "../tasks/task" + task_index + "/train" + str(group_index) + ".pickle",
                           'item_num': info['#node'],
                           'embedding_path': root_path + "/embedding" + str(group_index) + ".pickle",
                           'trans_matrix_path': root_path + "/trans_matrix" + str(group_index) + ".pickle"})
        result = pool.apply_async(run, args=[parameters])
        results.append(result)
    for res in results:
        res.get()
    log_file.close()
