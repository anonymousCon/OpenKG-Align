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
    np.random.shuffle(triple)
    return triple


# data_dir = 'SRPRS/dbp_yg' #DBP15K/zh_en
data_dir = 'SRPRS/en_fr'
# data_dir = 'fb_dbp'

ins2id_dict, id2ins_dict, [kg1_ins_ids, kg2_ins_ids] = load_dict(data_dir + "/ent_ids_", file_num=2)
triple1_idx = load_triples(data_dir + "/triples_1", file_num=1)
triple2_idx = load_triples(data_dir + "/triples_2", file_num=1)


G1 = nx.Graph()

for entid in kg1_ins_ids:
    G1.add_node(entid)

for item in triple1_idx:
    G1.add_edge(item[0], item[2])

print("Nodes in G1: " + str(len(G1.nodes())))
print("Edges in G1: " + str(len(G1.edges())))
# nx.draw(G1)

G2 = nx.Graph()
for entid in kg2_ins_ids:
    G2.add_node(entid)

for item in triple2_idx:
    G2.add_edge(item[0], item[2])

print("Nodes in G2: " + str(len(G2.nodes())))
print("Edges in G2: " + str(len(G2.edges())))


ent2degree = dict()
for item in (triple1_idx):
    ent1 = item[0]
    if ent1 not in ent2degree:
        ent2degree[ent1] = 1
    else:
        ent2degree[ent1] += 1
    ent1 = item[2]
    if ent1 not in ent2degree:
        ent2degree[ent1] = 1
    else:
        ent2degree[ent1] += 1

ouf = open(data_dir + '/degree_1.txt', 'w')
for item in ent2degree.keys():
    ouf.write(str(item) + '\t' + str(ent2degree[item]) + '\n')

ent2degree = dict()
for item in (triple2_idx):
    ent1 = item[0]
    if ent1 not in ent2degree:
        ent2degree[ent1] = 1
    else:
        ent2degree[ent1] += 1
    ent1 = item[2]
    if ent1 not in ent2degree:
        ent2degree[ent1] = 1
    else:
        ent2degree[ent1] += 1

ouf = open(data_dir + '/degree_2.txt', 'w')
for item in ent2degree.keys():
    ouf.write(str(item) + '\t' + str(ent2degree[item]) + '\n')

ouf = open(data_dir + '/pr_1.txt', 'w')
bet1 = nx.pagerank(G1)
print(bet1)
for item in bet1.items():
    ouf.write(str(item[0]) + '\t' + str(item[1]) + '\n')

ouf = open(data_dir + '/pr_2.txt', 'w')
bet1 = nx.pagerank(G2)
for item in bet1.items():
    ouf.write(str(item[0]) + '\t' + str(item[1]) + '\n')