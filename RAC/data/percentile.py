import networkx as nx
import numpy as np


def load_dict(data_dir, file_num=2):
    if file_num == 2:
        file_names = [data_dir + str(i) for i in range(1, 3)]
    else:
        file_names = [data_dir]
    what2id, id2what, ids = {}, {}, []
    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split("\t") for i in data]
            what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
            id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
            ids.append(set([int(i[0]) for i in data]))
    return what2id, id2what, ids

def load_triples(data_dir, file_num=2):
    if file_num == 2:
        file_names = [data_dir + str(i) for i in range(1, 3)]
    else:
        file_names = [data_dir]
    triple = []
    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [tuple(map(int, i.split("\t"))) for i in data]
            triple += data
    return triple


# data_dir = 'SRPRS/dbp_yg' #DBP15K/zh_en SRPRS/en_de fb_dbp
data_dir = 'SRPRS/en_fr'


ins2id_dict, id2ins_dict, [kg1_ins_ids, kg2_ins_ids] = load_dict(data_dir + "/ent_ids_", file_num=2)
triple1_idx = load_triples(data_dir + "/triples_1", file_num=1)
triple2_idx = load_triples(data_dir + "/triples_2", file_num=1)
ill_idx = load_triples(data_dir + "/ill_ent_ids", file_num=1)

ill_train_idx = np.array(ill_idx[int(len(ill_idx) // 1 * (1-0.3)):], dtype=np.int32)

metric_name = 'pr' #'pr'

# convert the scores to percentile
train_left = []
train_right = []
for item in ill_train_idx:
    train_left.append(item[0])
    train_right.append(item[1])

id2score = dict()
inf = open(data_dir + '/' + metric_name + '_1.txt')  # betwnorm_1 eigen_1
scores1 = []
for line in inf:
    print(line)
    strs = line.strip().split('\t')
    if int(strs[0]) in train_left:
        id2score[int(strs[0])] = float(strs[1])
        scores1.append(float(strs[1]))
scores1.sort(reverse=True)
score2perc = dict()
for i in range(len(scores1)):
    score2perc[scores1[i]] = (len(scores1) - i + 1) * 1.0 / len(scores1)

id2perc = dict()
for item in train_left:
    id2perc[item] = score2perc[id2score[item]]

inf = open(data_dir + '/' + metric_name + '_2.txt')
scores2 = []
for line in inf:
    strs = line.strip().split('\t')
    if int(strs[0]) in train_right:
        id2score[int(strs[0])] = float(strs[1])
        scores2.append(float(strs[1]))
scores2.sort(reverse=True)
score2perc = dict()
for i in range(len(scores2)):
    score2perc[scores2[i]] = (len(scores2) - i + 1) * 1.0 / len(scores2)

for item in train_right:
    id2perc[item] = score2perc[id2score[item]]

ouf = open(data_dir + '/'+metric_name+'_perc.txt', 'w')
for item in id2perc.keys():
    ouf.write(str(item) + '\t' + str(id2perc[item]) + '\n')
