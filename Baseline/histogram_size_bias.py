# this is test for histogram+size bias
import pickle
from utility.metric import l1_distance
from utility.util import *
import random


# histogram_size_bias(train, sizes, predicting_size)
# train is the training data
# sizes is the size distribution
# predicting_size is the size of predicting data
# it will return a predicting set, it is a python list like [{1}, {2, 5, 7}, {1, 3}, ...]
def histogram_size_bias(train, sizes, predicting_size):
    k = len(sizes)  # we won't generate subset whose size > k

    data = []  # data[i] is the list who contains all size i-1 subset
    for i in range(k):
        data.append([])
    for subset in train:
        l = len(subset)
        if l <= k:
            data[l - 1].append(subset)

    predict = []
    for i in range(k):
        predict.extend(random.choices(data[i], k=int(predicting_size * sizes[i])))
    return predict


# histogram(groupNum, train_paths, test_paths, log_path, metric)
# groupNum is how many (train, test) group you want to tet
# train_paths, test_paths and predict_path are the training, testing and predicting data path in list
# predicting_size is the size of predicting data
# size_paths is the size distribution path in list
# log_path is the output log.txt file path
# metric is the metric you want to evaluate histogram of training data with testing data
def histogram_size_bias_for_group(groupNum, train_paths, test_paths, predict_paths, predicting_size, size_paths, log_path, metric):
    log_file = open(log_path, "w")
    log_file.write("We use histogram + size bias to predict.\n")
    log_file.write("There are "+str(groupNum)+" groups.\n")
    log_file.write("We will predict "+str(predicting_size)+" subsets.\n")
    log_file.write("The bias size we use is from "+size_paths[0]+" and so on.\n")
    log_file.write("The metric we use is "+metric.__name__+".\n")
    log_file.write("The result of each group as fellow:\n")
    for group_index in range(groupNum):
        train = pickle.load(open(train_paths[group_index], 'rb'))
        sizes = pickle.load(open(size_paths[group_index], 'rb'))
        predict = histogram_size_bias(train, sizes, predicting_size)
        pickle.dump(predict, open(predict_paths[group_index], 'wb'))
        test = pickle.load(open(test_paths[group_index], 'rb'))
        result = metric(test, predict)
        log_file.write(str(result)+"\n")
    log_file.close()


# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
size_bias_root = "../tasks/task"+task_index+"/SizeBias/plan1/Tue Apr 12 14_47_01 2022"
output_size = 10000000
metric = l1_distance
# -----------------------------------------------------------------------------------------------------

info = pickle.load(open("../tasks/task" + task_index + "/info.pickle", "rb"))

time_str = timestamp()
time_str = time_str.replace(':', '_')
mkdir("../tasks/task"+task_index+"/Baseline")
mkdir("../tasks/task"+task_index+"/Baseline/histogram+size_bias/")
mkdir("../tasks/task"+task_index+"/Baseline/histogram+size_bias/"+time_str)

root_path = "../tasks/task"+task_index+"/Baseline/histogram+size_bias/"+time_str
log_path = root_path + "/log.txt"


train_paths = []
test_paths = []
predict_paths = []
size_paths = []
for group_index in range(int(info['#group'])):
    train_paths.append("../tasks/task"+task_index+"/train"+str(group_index)+".pickle")
    test_paths.append("../tasks/task"+task_index+"/test"+str(group_index)+".pickle")
    predict_paths.append(root_path + "/predict"+str(group_index)+".pickle")
    size_paths.append(size_bias_root+"/bias_size"+str(group_index)+".pickle")

histogram_size_bias_for_group(groupNum=int(info['#group']),
                              train_paths=train_paths,
                              test_paths=test_paths,
                              predict_paths=predict_paths,
                              predicting_size=output_size,
                              size_paths=size_paths,
                              log_path=log_path,
                              metric=metric)