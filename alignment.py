import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

# data1 = [95.51, 95.49, 95.45, 95.59, 95.56]
# data2 = [95.63, 95.58, 95.58, 95.62, 95.57]
#
# stat_val, p_val = stats.ttest_ind(data1, data2, equal_var=False)
#
# print(stat_val)
# print(p_val)


def load_tree(name):
    datasets = []
    with open(name, 'r') as f:
        sent = []
        data = []
        for line in f.readlines():
            if len(line.strip()) == 0:
                datasets.append((' '.join(sent), data))
                sent = []
                data = []

            else:
                data.append(line.strip())
                sent.append(line.strip().split('\t')[1])
    return datasets


def load_sent(name):
    dataset = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = line.strip().replace('\t', ' ')
            dataset.append(line)
    return dataset


def write_data(name, dataset):
    with open(name, 'w') as f:
        for data in dataset:
            for line in data:
                f.write(line)
                f.write('\n')
            f.write('\n')


filepath1 = '/media/zhanglw/2EF0EF5BF0EF27B3/code/parsing_retrain_human_eval/human_3_tree.conllu'
filepath2 = '/media/zhanglw/2EF0EF5BF0EF27B3/code/parsing_retrain_human_eval/base_b16_unk1000_parserbb_0pred_test_parseA.txt'

file1 = load_tree(filepath1)
file2 = load_tree(filepath2)

res = []
for i in range(len(file1)):
    for j in range(len(file2)):
        if file1[i][0] == file2[j][0]:
            res.append(file2[j][1])
        else:
            continue
write_data(filepath2 + '_reshape', res)
