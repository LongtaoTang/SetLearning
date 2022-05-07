
# this is test for histogram
import pickle
from utility.metric import l1_distance
from utility.util import *


# histogram(groupNum, train_paths, test_paths, log_path, metric)
# groupNum is how many (train, test) group you want to tet
# train_paths and test_paths are the training and testing data path in list
# log_path is the output log.txt file path
# metric is the metric you want to evaluate histogram of training data with testing data
def histogram(groupNum, train_paths, test_paths, log_path, metric):   # checked
    log_file = open(log_path, "w")
    log_file.write("We use histogram to predict.\n")
    log_file.write("There are "+str(groupNum)+" groups.\n")
    log_file.write("The metric we use is "+metric.__name__+".\n")
    log_file.write("The result of each group as fellow:\n")
    for group_index in range(groupNum):
        predict = pickle.load(open(train_paths[group_index], 'rb'))
        test = pickle.load(open(test_paths[group_index], 'rb'))
        result = metric(test, predict)
        log_file.write(str(result)+"\n")
    log_file.close()


# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
# -----------------------------------------------------------------------------------------------------

f = open("../tasks/task"+task_index+"/info.pickle", "rb")
info = pickle.load(f)
f.close()

time_str = timestamp()
time_str = time_str.replace(':', '_')
mkdir("../tasks/task"+task_index+"/Baseline")
mkdir("../tasks/task"+task_index+"/Baseline/histogram/")
mkdir("../tasks/task"+task_index+"/Baseline/histogram/"+time_str)

log_path = "../tasks/task"+task_index+"/Baseline/histogram/"+time_str+"/log.txt"


train_paths = []
test_paths = []
for group_index in range(int(info['#group'])):
    train_paths.append("../tasks/task"+task_index+"/train"+str(group_index)+".pickle")
    test_paths.append("../tasks/task"+task_index+"/test"+str(group_index)+".pickle")

histogram(int(info['#group']), train_paths, test_paths, log_path, l1_distance)
