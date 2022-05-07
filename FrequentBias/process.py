# frequent_bias_plan1(data, threshold, remain_p)
# data is the data set, it should be a python list like [{1}, {2, 5, 7}, {1, 3}, ...]
# threshold is the threshold to judge a subset is a frequent set or not
# If threshold > 1, we judge the subset by it's count larger than threshold or not
# If 1 > threshold > 0, we judge the subset by it's proportion over data larger or equal than threshold or not
# If a subset is not a frequent set, remain_count = int(remain_p * original_count).
# It will return a processed data list, Noticed we won't change the original data
def frequent_bias_plan1(data, threshold, remain_p):   # checked
    set_dict = dict()
    for subset in data:
        subset = list(subset)
        subset.sort()
        subset = tuple(subset)
        if subset not in set_dict:
            set_dict[subset] = 1
        else:
            set_dict[subset] += 1

    predict = []
    for subset_tuple in set_dict.keys():
        count = set_dict[subset_tuple]
        subset = set(subset_tuple)
        if threshold > 1:
            # If threshold > 1, we judge the subset by it's count larger or equal than threshold or not
            if count >= threshold:
                for i in range(int(count)):
                    predict.append(subset)
            else:
                for i in range(int(count * remain_p)):
                    predict.append(subset)
        elif threshold > 0:
            # If threshold in [0, 1], we judge by it's proportion over data larger or equal than threshold or not
            if count/len(data) >= threshold:
                for i in range(int(count)):
                    predict.append(subset)
            else:
                for i in range(int(count * remain_p)):
                    predict.append(subset)
        else:
            print("error threshold")
            exit(1)
    return predict


