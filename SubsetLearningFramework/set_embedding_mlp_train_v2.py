from SetEmbeddingv2 import *
import pickle
from multiprocessing import Pool
from utility.util import *
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------------------------------
# need change
task_index = "3"
model_class = SetEmbeddingModelV2
embedding_dim = 10
batch_size = 100
num_of_samples = 50
num_of_training = 4
optimizer_method = torch.optim.RMSprop
graph_root = "../tasks/task" + str(task_index) + "/SparseGraph/Sun Apr 10 00_12_10 2022"


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


def run(parameters):
    data = pickle.load(open(parameters['data_path'], 'rb'))
    node_num = parameters['node_num']
    graph = pickle.load(open(parameters['graph_path'], 'rb'))

    net = SetNet()
    model = model_class(node_num=node_num, embedding_dim=embedding_dim,
                        graph=graph, embedding_func=None,
                        embedding_func_parameters=net.parameters(),
                        net=net)

    optimizer = optimizer_method(params=model.parameters)

    for train_index in range(num_of_training):
        # train a round
        model.train(data, batch_size=batch_size,
                    num_of_samples=num_of_samples,
                    num_of_training=1,
                    optimizer=optimizer)
        # save model
        f = open(parameters['save_root'] + "/model" + str(train_index) + ".pickle", 'wb')
        pickle.dump(model, f)
        f.close()


if __name__ == '__main__':
    info = pickle.load(open("../tasks/task" + task_index + "/info.pickle", "rb"))

    time_str = timestamp()
    time_str = time_str.replace(':', '_')
    mkdir("../tasks/task" + task_index + "/SLF")
    mkdir("../tasks/task" + task_index + "/SLF/" + model_class.__name__)
    mkdir("../tasks/task" + task_index + "/SLF/" + model_class.__name__ + "/" + time_str)

    root_path = "../tasks/task" + task_index + "/SLF/" + model_class.__name__ + "/" + time_str
    log_file = open(root_path + "/log_train.txt", 'w')
    log_file.write("we use " + model_class.__name__ + " for training." + "\n")
    log_file.write("embedding_func: mlp\n")
    log_file.write("embedding dim: " + str(embedding_dim) + "\n")
    log_file.write("batch_size: " + str(batch_size) + "\n")
    log_file.write("num_of_samples: " + str(num_of_samples) + "\n")
    log_file.write("num_of_training: " + str(num_of_training) + "\n")
    log_file.write("optimizer_method: " + optimizer_method.__name__ + "\n")
    log_file.write("graph_root: " + graph_root + "\n")
    log_file.write("In each dictionary, we put the model after train of each group. " +
                   "The model5.pickle means the model was train by 5+1 round of whole data set.\n")
    log_file.close()
    pool = Pool(8)
    results = []
    for group_index in range(int(info['#group'])):
        mkdir(root_path + "/" + str(group_index))
        parameters = dict({'data_path': "../tasks/task" + task_index + "/train" + str(group_index) + ".pickle",
                           'save_root': root_path + "/" + str(group_index),
                           'node_num': info['#node'],
                           'graph_path': graph_root + "/graph" + str(group_index) + ".pickle"})
        result = pool.apply_async(run, args=[parameters])
        results.append(result)
    for res in results:
        res.get()
