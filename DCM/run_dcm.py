from utility.metric import l1_distance
from utility.util import *
import pickle
import numpy as np
from multiprocessing import Pool
from DCM import DCM

# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
size_bias_root = "../tasks/task" + task_index + "/SizeBias/plan1/Tue Apr 12 14_47_01 2022"
use_size_bias = True
output_size = 10000000
metric = l1_distance
process_size = 5  # should >= 5
h_sizes = np.ones(process_size) * 100


# -----------------------------------------------------------------------------------------------------


def run(parameters):
    data = pickle.load(open(parameters['data_path'], 'rb'))
    dcm = DCM(parameters['process_size'], parameters['h_sizes'])
    dcm.train(data, parameters['item_num'])
    if parameters['use_size_bias']:
        predict = dcm.predict(parameters['output_size'],
                              size_distribution=pickle.load(open(parameters['size_bias_path'], 'rb')))
    else:
        predict = dcm.predict(parameters['output_size'])
    pickle.dump(predict, open(parameters['predict_path'], 'wb'))
    test_data = pickle.load(open(parameters['test_path'], 'rb'))
    return parameters['metric'](predict, test_data)


if __name__ == '__main__':
    info = pickle.load(open("../tasks/task" + task_index + "/info.pickle", "rb"))

    time_str = timestamp()
    time_str = time_str.replace(':', '_')
    mkdir("../tasks/task" + task_index + "/DCM")
    mkdir("../tasks/task" + task_index + "/DCM/" + time_str)

    root_path = "../tasks/task" + task_index + "/DCM/" + time_str
    log_path = root_path + "/log.txt"

    log_file = open(log_path, 'w')
    log_file.write("we use DCM method." + "\n")
    log_file.write("use_size_bias: " + str(use_size_bias) + "\n")
    log_file.write("size_bias_root: " + size_bias_root + "\n")
    log_file.write("output_size: " + str(output_size) + "\n")
    log_file.write("metric: " + metric.__name__ + "\n")
    log_file.write("DCM process_size: " + str(process_size) + "\n")
    log_file.write("DCM h_sizes: " + str(h_sizes) + "\n")
    log_file.write("There are " + str(info['#group']) + " data" + "\n")
    log_file.write("results are:" + "\n")

    pool = Pool(8)
    results = []
    for group_index in range(int(info['#group'])):
        parameters = dict({'data_path': "../tasks/task" + task_index + "/train" + str(group_index) + ".pickle",
                           'root_path': root_path,
                           'output_size': output_size,
                           'use_size_bias': use_size_bias,
                           'size_bias_path': size_bias_root + "/bias_size" + str(group_index) + ".pickle",
                           'metric': l1_distance,
                           'h_sizes': h_sizes,
                           'process_size': process_size,
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
