

def l1_distance(data_real, data_predict):
    N = len(data_real)
    M = len(data_predict)

    predict_dict = dict()
    compare_dict = dict()

    for subset in data_predict:
        subset = list(subset)
        subset.sort()
        subset = tuple(subset)
        if subset not in predict_dict:
            predict_dict[subset] = 1
            compare_dict[subset] = 0
        else:
            predict_dict[subset] += 1

    for subset in data_real:
        subset = list(subset)
        subset.sort()
        subset = tuple(subset)
        if subset in compare_dict:
            compare_dict[subset] += 1

    overleap = 0
    for key, value in predict_dict.items():
        overleap += min(value/M, compare_dict[key]/N)
    l1 = 2 - 2 * overleap
    # print(l1)
    return l1
