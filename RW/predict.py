import pickle
from multiprocessing import Pool
from utility.metric import l1_distance
from utility.util import *
import numpy as np
import random

# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
train_results_root = "../tasks/task" + task_index + "/RW/Tue Apr 12 14_57_44 2022"
size_bias_root = "../tasks/task" + task_index + "/SizeBias/plan1/Tue Apr 12 14_47_01 2022"
use_size_bias = False
output_size = 10000000
metric = l1_distance
# -----------------------------------------------------------------------------------------------------


def generate(cum_trans_matrix, early_stop):
    node_num = cum_trans_matrix.shape[0]

    subset = set({})

    x = 0

    while True:
        y = random.choices(range(node_num), cum_weights=cum_trans_matrix[x], k=1)[0]
        x = y
        if x == node_num - 1:
            break
        else:
            subset.add(x)
        if len(subset) > early_stop:
            # we give another try
            subset = set({})
            x = 0
    return subset


def run(parameters):
    trans_matrix = pickle.load(open(parameters['trans_matrix_path'], 'rb'))
    cum_trans_matrix = np.zeros(trans_matrix.shape)
    for row in range(trans_matrix.shape[0]):
        sum_p = 0
        for col in range(trans_matrix.shape[1]):
            sum_p += trans_matrix[row][col]
            cum_trans_matrix[row][col] = sum_p

    predict = list([])
    if parameters['use_size_bias']:
        print("we use size bias")
        bias_size = pickle.load(open(parameters['size_bias_path'], 'rb'))
        target_bucket = np.zeros(len(bias_size))
        current_bucket = np.zeros(len(bias_size))
        for i in range(len(bias_size)):
            target_bucket[i] = int(parameters['output_size'] * bias_size[i])
        print("target_bucket: ", target_bucket)

        early_stop = len(bias_size)
        total_count = 0
        while (early_stop > 0):
            subset = generate(cum_trans_matrix, early_stop)
            size = len(subset) - 1
            if current_bucket[size] >= target_bucket[size]:
                # we reject
                pass
            else:
                predict.append(subset)
                total_count += 1
                current_bucket[size] += 1
                if total_count % 1000000 == 0:
                    print(total_count)
                    print(current_bucket)
                    print(target_bucket)
            if current_bucket[early_stop - 1] >= target_bucket[early_stop - 1]:
                early_stop = early_stop - 1
                print("early_stop: ", early_stop)
    else:
        early_stop = trans_matrix.shape[0] - 2
        for i in range(output_size):
            subset = generate(cum_trans_matrix, early_stop)
            predict.append(subset)
            if i % 1000000 == 0:
                print(i)

    pickle.dump(predict, open(parameters['predict_path'], 'wb'))
    test = pickle.load(open(parameters['test_path'], 'rb'))
    return parameters['metric'](test, predict)


if __name__ == '__main__':
    info = pickle.load(open("../tasks/task" + task_index + "/info.pickle", "rb"))

    time_str = timestamp()
    time_str = time_str.replace(':', '_')
    mkdir(train_results_root + "/" + time_str)
    root_path = train_results_root + "/" + time_str
    log_path = root_path + "/log.txt"
    log_file = open(log_path, 'w')
    log_file.write("we predict by RW."+"\n")
    log_file.write("use_size_bias: "+str(use_size_bias)+"\n")
    log_file.write("size_bias_root: " + size_bias_root + "\n")
    log_file.write("output_size: "+str(output_size)+"\n")
    log_file.write("metric: " + metric.__name__ + "\n")
    log_file.write("There are " + str(info['#group']) + " data" + "\n")
    log_file.write("results are:" + "\n")

    pool = Pool(8)
    results = []
    for group_index in range(int(info['#group'])):
        parameters = dict({'trans_matrix_path': train_results_root + "/trans_matrix" + str(group_index) + ".pickle",
                           'output_size': output_size,
                           'use_size_bias': use_size_bias,
                           'size_bias_path': size_bias_root + "/bias_size" + str(group_index) + ".pickle",
                           'metric': metric,
                           'item_num': info['#node'],
                           'predict_path': root_path + "/predict" + str(group_index) + ".pickle",
                           'test_path': "../tasks/task" + task_index + "/test" + str(group_index) + ".pickle"})
        result = pool.apply_async(run, args=[parameters])
        results.append(result)
    for res in results:
        answer = res.get()
        log_file.write(str(answer) + "\n")
        print(answer)
    log_file.close()
