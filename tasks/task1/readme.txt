This task contains 8 training data and 8 testing data.

There are 1363 items in the set.
"info.pickle" is a python dictionary store all information of this task, You should use pickle.load() to load it.
info = dict({'#node': 1363, '#group': 8, '#train': train_size, '#test': test_size})

The dataset is about online shopping order in Tmall.
We split each data in a month by a training set and a testing set.
Each training or testing set is a python list like "[{1}, {1, 3}, {2, 5, 9}, ...]".
Each element in list is a python set.
You should use pickle.load() to load those files.
The size, month information as fellow.

group: 0
month: 201808
total size: 428343
train size: 100000
test size: 328343

group: 1
month: 201809
total size: 484444
train size: 100000
test size: 384444

group: 2
month: 201810
total size: 548349
train size: 100000
test size: 448349

group: 3
month: 201811
total size: 668801
train size: 100000
test size: 568801

group: 4
month: 201812
total size: 587095
train size: 100000
test size: 487095

group: 5
month: 201901
total size: 480005
train size: 100000
test size: 380005

group: 6
month: 201902
total size: 292331
train size: 100000
test size: 192331

group: 7
month: 201903
total size: 456438
train size: 100000
test size: 356438

The unsplit python list are stored in dictionary "unsplit data", they have been shuffled and training set is the first 100000 subset in unsplit data.

The source of this dataset is https://github.com/wtw666/Representation-Learning-for-Predicting-Customer-Orders.
