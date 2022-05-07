from SetEmbedding import *
import pickle
from multiprocessing import Pool
from utility.util import *
from utility.metric import l1_distance
import torch
import torch.nn as nn

embedding_dim = 10
# -----------------------------------------------------------------------------------------------------
# net
class SetNet(nn.Module):
    def __init__(self):
        super(SetNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 5)
        self.fc2 = nn.Linear(embedding_dim * 5, embedding_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------------------------------
# need change
task_index = "3"
train_root = "../tasks/task" + str(task_index) + "/SLF/SetEmbeddingModelV2/Fri Apr 29 16_02_44 2022"
train_index = 3
size_bias_root = "../tasks/task" + str(task_index) + "/SizeBias/plan1/Sun Apr 10 00_09_49 2022"
use_size_bias = False
output_size = 10000000
metric = l1_distance
# -----------------------------------------------------------------------------------------------------


def run(parameters):
    f = open(parameters['train_path'], 'rb')
    model = pickle.load(f)
    f.close()
    if use_size_bias:
        print("we use size bias")
        bias_size = pickle.load(open(parameters['size_bias_path'], 'rb'))
        predict = model.predict(output_size=output_size, bias_size=bias_size)
    else:
        predict = model.predict(output_size=output_size)

    pickle.dump(predict, open(parameters['predict_path'], 'wb'))
    test = pickle.load(open(parameters['test_path'], 'rb'))
    return parameters['metric'](test, predict)


if __name__ == '__main__':
    info = pickle.load(open("../tasks/task" + str(task_index) + "/info.pickle", "rb"))

    time_str = timestamp()
    time_str = time_str.replace(':', '_')
    mkdir(train_root + "/predict")
    mkdir(train_root + "/predict/" + time_str)

    root_path = train_root + "/predict/" + time_str
    log_file = open(root_path + "/log_predict.txt", 'w')
    log_file.write("use_size_bias: " + str(use_size_bias) + '\n')
    log_file.write("size_bias_root: " + str(size_bias_root) + '\n')
    log_file.write("output_size: " + str(output_size) + '\n')
    log_file.write("metric: " + metric.__name__ + '\n')
    log_file.write("There are " + str(info['#group']) + " data" + "\n")
    log_file.write("results are:" + "\n")

    pool = Pool(8)
    results = []
    for group_index in range(int(info['#group'])):
        mkdir(root_path + "/" + str(group_index))
        parameters = dict({'train_path': train_root + "/" + str(group_index) + "/model" + str(train_index) + ".pickle",
                           'predict_path': root_path + "/predict" + str(group_index) + ".pickle",
                           'test_path': "../tasks/task" + str(task_index) + "/test" + str(group_index) + ".pickle",
                           'metric': metric,
                           'size_bias_path': size_bias_root + "/bias_size" + str(group_index) + ".pickle",
                           })
        result = pool.apply_async(run, args=[parameters])
        results.append(result)
    for res in results:
        answer = res.get()
        log_file.write(str(answer) + "\n")
        print(answer)
    log_file.close()
