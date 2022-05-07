def mkdir(path):
    import os

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def timestamp():
    import time
    return time.asctime(time.localtime(time.time()))


def txt_to_pickle(f):
    data = list()

    for line in f.readlines():
        line = line[:-2]
        line = line.split(" ")
        subset = set()
        for x in line:
            x = int(x)
            subset.add(x)
        data.append(subset)

    return data
