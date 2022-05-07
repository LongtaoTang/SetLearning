import numpy as np
import pickle
from utility.util import timestamp, mkdir

# Let sizes = [a_0, a_1, a_2, a_3, a_4, a_5, ...] where ai represent the proportion of size i-1 subset.
# we first calculate b_i = a_i / (\sum_{j>=i} a_j)
# Then we have a_i = (1-b_0)(1-b_1)...(1-b_{i-1})b_i
# We modify b_i to c_i, where c_i = b_i + (i+1) sqrt(#item / #training data)
# Then we got the modified size distribution a'_i = (1-c_0)(1-c_1)...(1-c_{i-1})c_i
# Especially, we set c_k = 1, which means we do not generate any size > k subset, here k is user define
# bias_size_path well shore the bias size, which is a numpy array in pickle
# log_path well shore the log.txt, with information of this bias size
# it will return a numpy array which is the biased size distribution


def size_bias_plan1(bias_size_path, log_path, training_date_path, item_number, k):
    f = open(training_date_path, 'rb')
    data = pickle.load(f)
    f.close()

    a = np.zeros(k)
    for subset in data:
        l = len(subset)
        if l <= k:
            a[l-1] += 1
    for i in range(k):
        a[i] = a[i] / len(data)
    print(a)

    b = np.zeros(k)
    rest = 1
    for i in range(k):
        b[i] = a[i] / rest
        rest = rest - a[i]
    print(b)

    c = np.zeros(k)
    for i in range(k):
        c[i] = b[i] + (i+1) * np.sqrt(item_number / len(data))
        if c[i] > 1:
            c[i] = 1
    print(c)

    d = np.zeros(k)
    rest = 1
    for i in range(k):
        d[i] = rest * c[i]
        rest = rest - d[i]
    print(d)

    f = open(bias_size_path, 'wb')
    pickle.dump(d, f)
    f.close()

    f = open(log_path, 'w')
    f.write("We use size_bias_plan1.\n")
    f.write("k = " + str(k) + "\n")
    f.write("we do not generate subset with size >=" + str(k) + "\n")
    f.write("The size of training data is:\n")
    f.write(str(a)+"\n")
    f.write("The over rest coefficient is:\n")
    f.write(str(b) + "\n")
    f.write("The modified over rest coefficient is:\n")
    f.write(str(c) + "\n")
    f.write("The biased size is:\n")
    f.write(str(d) + "\n")
    f.close()

    return d


# -----------------------------------------------------------------------------------------------------
# need change
task_index = "4"
k = 5
# -----------------------------------------------------------------------------------------------------

f = open("../tasks/task"+task_index+"/info.pickle", "rb")
info = pickle.load(f)
f.close()

time_str = timestamp()
time_str = time_str.replace(':', '_')
mkdir("../tasks/task"+task_index+"/SizeBias")
mkdir("../tasks/task"+task_index+"/SizeBias/plan1/")
mkdir("../tasks/task"+task_index+"/SizeBias/plan1/"+time_str)

pre_path = "../tasks/task"+task_index+"/SizeBias/plan1/"+time_str

for group_index in range(int(info['#group'])):
    size_bias_plan1(bias_size_path=pre_path+"/bias_size"+str(group_index)+".pickle",
                    log_path=pre_path+"/log"+str(group_index)+".txt",
                    training_date_path="../tasks/task"+task_index+"/train"+str(group_index)+".pickle",
                    item_number=int(info['#node']),
                    k=k)
