#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    run.py
  Author:       locke
  Date created: 2020/3/25 下午6:58
"""

import time
import argparse
import os
import gc
import random
import math
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from load_data import *
from models import *
from utils import *
import copy

# from torch.utils.tensorboard import SummaryWriter
# import logging
from sklearn.cluster import KMeans
import scipy


class Experiment:
    def __init__(self, args):
        self.vali = False
        self.save = args.save
        self.save_prefix = "%s_%s" % (args.data_dir.split("/")[-1], args.log)

        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))

        self.args = args
        self.args.encoder = args.encoder.lower()
        self.args.encoder1 = args.encoder1.lower()

        self.args.decoder = args.decoder.lower()
        self.args.sampling = args.sampling
        self.args.k = int(args.k)
        self.args.margin = float(args.margin)
        self.args.alpha = float(args.alpha)

        ##ent pairs
        self.lefts_test = [i[0] for i in d.ill_test_idx]
        self.rights_test = [i[1] for i in d.ill_test_idx]

        self.lefts_train = [i[0] for i in d.ill_train_idx]
        self.rights_train = [i[1] for i in d.ill_train_idx]

        self.lefts = [i[0] for i in d.ill_idx]
        self.rights = [i[1] for i in d.ill_idx]

        if len(self.lefts) > 15000:
            self.lefts = self.lefts[len(self.lefts) - 15000:]
            self.rights = self.rights[len(self.rights) - 15000:]

        self.fc1 = torch.nn.Linear(self.hiddens[-1], self.hiddens[-1]).to(device)
        self.fc2 = torch.nn.Linear(self.hiddens[-1], self.hiddens[-1]).to(device)

        self.cached_sample = {}
        self.best_result = ()

    def evaluate(self, it, test, ins_emb, ins_emb1, mapping_emb=None, vali_flag= False):
        t_test = time.time()
        top_k = [1, 3, 5, 10]
        if mapping_emb is not None:
            print("using mapping")
            left_emb = mapping_emb[test[:, 0]]
        else:
            left_emb = ins_emb[test[:, 0]]
        right_emb = ins_emb[test[:, 1]]
        distance = - sim(left_emb, right_emb, metric=self.args.test_dist, normalize=True,
                         csls_k=self.args.csls)  # normalize = True.... False can increase performance

        if self.args.two_views == 1 and self.args.fuse_embed != 1:
            left_emb1 = ins_emb1[test[:, 0]]
            right_emb1 = ins_emb1[test[:, 1]]
            distance1 = - sim(left_emb1, right_emb1, metric=self.args.test_dist, normalize=True, csls_k=self.args.csls)
            distance = distance * self.args.alp + distance1 * (1 - self.args.alp)

        if self.args.rerank:
            indices = np.argsort(np.argsort(distance, axis=1), axis=1)
            indices_ = np.argsort(np.argsort(distance.T, axis=1), axis=1)
            distance = indices + indices_.T

        tasks = div_list(np.array(range(len(test))), 10)
        pool = multiprocessing.Pool(processes=len(tasks))
        reses = list()
        for task in tasks:
            reses.append(
                pool.apply_async(multi_cal_rank, (task, distance[task, :], distance[:, task], top_k, self.args)))
        pool.close()
        pool.join()

        acc_l2r, acc_r2l = np.array([0.] * len(top_k)), np.array([0.] * len(top_k))
        mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0., 0., 0., 0.
        for res in reses:
            (_acc_l2r, _mean_l2r, _mrr_l2r, _acc_r2l, _mean_r2l, _mrr_r2l) = res.get()
            acc_l2r += _acc_l2r
            mean_l2r += _mean_l2r
            mrr_l2r += _mrr_l2r
            acc_r2l += _acc_r2l
            mean_r2l += _mean_r2l
            mrr_r2l += _mrr_r2l
        mean_l2r /= len(test)
        mean_r2l /= len(test)
        mrr_l2r /= len(test)
        mrr_r2l /= len(test)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / len(test), 4)
            acc_r2l[i] = round(acc_r2l[i] / len(test), 4)

        if vali_flag is False:
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r.tolist(),
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l.tolist(),
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))
        return (acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l)

    def init_emb(self):
        e_scale, r_scale = 1, 1
        self.ins_embeddings = nn.Embedding(d.ins_num, self.hiddens[0] * e_scale).to(device)
        self.rel_embeddings = nn.Embedding(d.rel_num, int(self.hiddens[0] * r_scale)).to(device)

        nn.init.xavier_normal_(self.ins_embeddings.weight)
        nn.init.xavier_normal_(self.rel_embeddings.weight)

        self.enh_ins_emb = self.ins_embeddings.weight.cpu().detach().numpy()
        self.mapping_ins_emb = None

    def prepare_input(self):
        graph_encoder = Encoder(self.args.encoder, self.hiddens, self.heads + [1], self.args.appkk, activation=F.elu,
                                feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2,
                                bias=False).to(device)

        # print(graph_encoder)

        knowledge_decoder = Decoder(self.args.decoder, params={
            "e_num": d.ins_num,
            "r_num": d.rel_num,
            "dim": self.hiddens[-1],
            "feat_drop": self.args.feat_drop,
            "train_dist": self.args.train_dist,
            "sampling": self.args.sampling,
            "k": self.args.k,
            "margin": self.args.margin,
            "alpha": self.args.alpha,
            "boot": self.args.bootstrap,
            # pass other useful parameters to Decoder
        }).to(device)
        # print(knowledge_decoder)

        train = np.array(d.ill_train_idx.tolist())
        np.random.shuffle(train)
        pos_batch = train
        neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                      [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                      params={"emb": self.enh_ins_emb, "metric": self.args.test_dist})

        if self.args.two_views == 1 and self.vali is False:
            graph_encoder1 = Encoder(self.args.encoder1, self.hiddens, self.heads + [1], self.args.appkk,
                                     activation=F.elu,
                                     feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2,
                                     bias=False).to(device)
            # print(graph_encoder1)

            return graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch
        else:
            return graph_encoder, knowledge_decoder, pos_batch, neg_batch

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def get_contrastive_loss(self, enh_emb, enh_emb1, temp=0.5):
        enh_emb = self.projection(enh_emb)
        enh_emb1 = self.projection(enh_emb1)
        f = lambda x: torch.exp(x / temp)

        refl_sim = f(self.sim(enh_emb, enh_emb))
        refl_sim_sum1 = refl_sim.sum(1)
        refl_sim_diag = refl_sim.diag()
        del refl_sim

        between_sim = f(self.sim(enh_emb, enh_emb1))
        between_sim_sum1 = between_sim.sum(1)
        between_sim_diag = between_sim.diag()
        del between_sim

        loss1 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

        refl_sim = f(self.sim(enh_emb1, enh_emb1))
        refl_sim_sum1 = refl_sim.sum(1)
        refl_sim_diag = refl_sim.diag()
        del refl_sim

        between_sim = f(self.sim(enh_emb1, enh_emb))
        between_sim_sum1 = between_sim.sum(1)
        between_sim_diag = between_sim.diag()
        del between_sim

        loss2 = -torch.log(between_sim_diag / (between_sim_sum1 + refl_sim_sum1 - refl_sim_diag))

        loss = (loss1.sum() + loss2.sum()) / (2 * len(enh_emb))

        # print(loss)
        return loss

    def get_loss(self, graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch, it):
        graph_encoder.train()
        knowledge_decoder.train()
        neg = torch.LongTensor(neg_batch).to(device)
        pos = torch.LongTensor(pos_batch).repeat(knowledge_decoder.k * 2, 1).to(device)
        use_edges = torch.LongTensor(d.ins_G_edges_idx).to(device)
        enh_emb = graph_encoder.forward(use_edges, self.ins_embeddings.weight)

        if self.args.two_views == 1 and self.vali is False:
            graph_encoder1.train()
            enh_emb1 = graph_encoder1.forward(use_edges, self.ins_embeddings.weight)
            enh_emb_final = enh_emb * self.args.alp + enh_emb1 * (1 - self.args.alp)

            # enh_emb_final = torch.cat((enh_emb, enh_emb1), dim=-1)
            if self.args.fuse_embed == 1:
                pos_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, neg)
                target = torch.ones(neg_score.size()).to(device)
                loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha
            else:
                pos_score = knowledge_decoder.forward(enh_emb, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb, self.rel_embeddings.weight, neg)
                target = torch.ones(neg_score.size()).to(device)
                loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

                pos_score = knowledge_decoder.forward(enh_emb1, self.rel_embeddings.weight, pos)
                neg_score = knowledge_decoder.forward(enh_emb1, self.rel_embeddings.weight, neg)
                target = torch.ones(neg_score.size()).to(device)
                loss1 = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

                loss = loss * self.args.alp + loss1 * (1 - self.args.alp)
                self.enh_emb = enh_emb.cpu().detach().numpy()
                self.enh_emb1 = enh_emb1.cpu().detach().numpy()

        else:
            enh_emb_final = enh_emb
            pos_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, pos)
            neg_score = knowledge_decoder.forward(enh_emb_final, self.rel_embeddings.weight, neg)
            target = torch.ones(neg_score.size()).to(device)
            loss = knowledge_decoder.loss(pos_score, neg_score, target) * knowledge_decoder.alpha

        self.enh_ins_emb = enh_emb_final.cpu().detach().numpy()  # fused embedding if two

        if self.args.two_views == 1 and self.args.contras_flag == 1 and self.vali is False:
            temperatue = 1
            left = enh_emb[self.lefts]
            left1 = enh_emb1[self.lefts]
            right = enh_emb[self.rights]
            right1 = enh_emb1[self.rights]
            loss1 = self.get_contrastive_loss(left, left1, temp=temperatue)
            loss1 += self.get_contrastive_loss(right, right1, temp=temperatue)
            # loss = loss + (0.1*loss2/2 + 0.1*loss1/2)#/2#/2
            loss = loss + 0.2 * loss1 / 2  # + 0.1 * loss2/2

        return loss

    def train_and_eval(self):
        self.init_emb()
        if self.args.two_views == 1:
            graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch = self.prepare_input()
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                      + [p for p in knowledge_decoder.parameters()]
                                      + (list(graph_encoder.parameters()))
                                      + (list(graph_encoder1.parameters())))

            if self.args.contras_flag == 1:
                params1 = nn.ParameterList((list(self.fc1.parameters())) + (list(self.fc2.parameters())))
                opt = optim.Adam([{'params': params}, {'params': params1, 'lr': 0.00001}], lr=self.args.lr,
                                 weight_decay=0.00001)
            else:
                opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        else:
            graph_encoder, knowledge_decoder, pos_batch, neg_batch = self.prepare_input()
            params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                      + [p for p in knowledge_decoder.parameters()]
                                      + (list(graph_encoder.parameters()))
                                      )
            opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        # print("Start training...")
        for it in range(0, self.args.epoch):
            t_ = time.time()
            opt.zero_grad()
            if self.args.two_views == 1:
                loss = self.get_loss(graph_encoder, graph_encoder1, knowledge_decoder, pos_batch, neg_batch, it)
            else:
                loss = self.get_loss(graph_encoder, None, knowledge_decoder, pos_batch, neg_batch,
                                     it)

            loss.backward()
            opt.step()
            loss = loss.item()
            loss_name = "loss_" + knowledge_decoder.print_name.replace("[", "_").replace("]", "_")

            if (it + 1) % self.args.update == 0:
                # logger.info("neg sampling...")
                neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                              [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                              params={"emb": self.enh_ins_emb,
                                                                      "metric": self.args.test_dist, })
            if self.vali is True:
                if (it + 1) % (300) == 0:
                    with torch.no_grad():
                        result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                        # H1 = result[0][0]
                        H1 = result[2]
                        break
            else:
                # Evaluate
                if (it + 1) % self.args.check == 0:
                    print("Start validating...")
                    with torch.no_grad():
                        if self.args.two_views == 1 and self.args.fuse_embed != 1:
                            result = self.evaluate(it, d.ill_test_idx, self.enh_emb, self.enh_emb1, self.mapping_ins_emb, self.vali)
                        else:
                            result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                    if it + 1 == self.args.epoch:
                        H1 = result[0][0]
                # self.best_result = result

        return self.enh_ins_emb, H1

    def train_and_eval_val(self):
        self.init_emb()

        graph_encoder, knowledge_decoder, pos_batch, neg_batch = self.prepare_input()
        params = nn.ParameterList([self.ins_embeddings.weight, self.rel_embeddings.weight]
                                  + [p for p in knowledge_decoder.parameters()]
                                  + (list(graph_encoder.parameters()))
                                  )
        opt = optim.Adam(params, lr=self.args.lr, weight_decay=0.00001)  # 0.00001

        # print("Start training...")
        for it in range(0, self.args.epoch):
            t_ = time.time()
            opt.zero_grad()
            loss = self.get_loss(graph_encoder, None, knowledge_decoder, pos_batch, neg_batch, it)

            loss.backward()
            opt.step()
            loss = loss.item()
            loss_name = "loss_" + knowledge_decoder.print_name.replace("[", "_").replace("]", "_")

            if (it + 1) % self.args.update == 0:
                # logger.info("neg sampling...")
                neg_batch = knowledge_decoder.sampling_method(pos_batch, d.triple_idx, d.ill_train_idx,
                                                              [d.kg1_ins_ids, d.kg2_ins_ids], knowledge_decoder.k,
                                                              params={"emb": self.enh_ins_emb,
                                                                      "metric": self.args.test_dist, })
            if (it + 1) % (300) == 0:
                with torch.no_grad():
                    result = self.evaluate(it, d.ill_test_idx, self.enh_ins_emb, None, self.mapping_ins_emb, self.vali)
                    # H1 = result[0][0]
                    H1 = result[2]
                    break

        return self.enh_ins_emb, H1

def perc(metric_name):
    id2perc = dict()
    inf = open(args.data_dir + '/' + metric_name + '_perc.txt')
    for line in inf:
        strs = line.strip().split('\t')
        id2perc[int(strs[0])] = float(strs[1])
    return id2perc

def score(metric_name):
    id2score = dict()
    inf = open(args.data_dir + '/' + metric_name + '_1.txt')
    for line in inf:
        strs = line.strip().split('\t')
        id2score[int(strs[0])] = float(strs[1])
    return id2score

def centrality_score(lefts, ablat):
    left2score = dict()
    for item in lefts:
        if ablat == '_degree':
            metric_name = 'degree'
            ent2value_deg = perc(metric_name)
            left2score[item] = ent2value_deg[item]
        elif ablat == '_pr':
            metric_name = 'pr'
            ent2value_pr = perc(metric_name)
            left2score[item] = ent2value_pr[item]
    return left2score

def information_den(enh_emb, lefts):
    train_embed = enh_emb[lefts]
    kmeans = KMeans().fit(train_embed)
    center_embeds = kmeans.cluster_centers_
    labels = kmeans.labels_
    ent2valueD = dict()
    scores = []
    for i in range(len(lefts)):
        emb = train_embed[i]
        dis = scipy.spatial.distance.euclidean(emb, center_embeds[int(labels[i])])
        ent2valueD[lefts[i]] = 1.0 / (1 + dis)
        scores.append(1.0 / (1 + dis))
    scores.sort(reverse=True)
    score2perc = dict()
    for i in range(len(scores)):
        score2perc[scores[i]] = (len(scores) - i + 1) * 1.0 / len(scores)
    left2score = dict()
    for item in lefts:
        left2score[item] = score2perc[ent2valueD[item]] # args.theta0
    return left2score

def update_dic_perc(lefts, score_dict):
    scores = []
    for i in range(len(lefts)):
        scores.append(score_dict[lefts[i]])
    scores.sort(reverse=True)
    score2perc = dict()
    for i in range(len(scores)):
        score2perc[scores[i]] = (len(scores) - i + 1) * 1.0 / len(scores)
    left2score = dict()
    for item in lefts:
        left2score[item] = score2perc[score_dict[item]]  # args.theta0
    return left2score

def suggesting_score(lefts, score_dicts, b, U, r, num_chosen):
    suggestedEnt2Score = dict()
    suggestedEnts = []
    # obtain the weight
    weights_reward = np.zeros(3)
    weights_explore = np.zeros(3)
    alpha = 0.5
    weights = np.zeros(3)
    aaa = 0.4
    bbb = 0.2
    if r <= 5:
        weights[0] = aaa; weights[1] = aaa; weights[2] = bbb
    else:
        for kkk in range(len(score_dicts)):
            weights_reward[kkk] = np.sum(U[kkk][:r - 1])*1.0/(r-1) # all the history...
            weights_explore[kkk] = math.sqrt(1.5 * math.log(r) / num_chosen[kkk])
        if np.sum(weights_reward) == 0:
            weights_reward[0] = 0.333; weights_reward[1] = 0.333; weights_reward[2] = 0.333
        else:
            weights_reward = weights_reward/np.sum(weights_reward) # normalize
        if np.sum(weights_explore) == 0:
            weights_explore[0] = 0.333; weights_explore[1] = 0.333; weights_explore[2] = 0.333
        else:
            weights_explore = weights_explore/np.sum(weights_explore) # normalize
        weights = alpha * weights_reward + (1 - alpha) * weights_explore
    # print('weight normalize')
    # print(weights)

    for kkk in range(len(score_dicts)):
        ranks = sorted(score_dicts[kkk].items(), key=lambda d: d[1], reverse=True)
        suggested = []
        for i in range(len(lefts)):
            id = ranks[i][0]
            suggested.append(id)
            if id not in suggestedEnt2Score:
                suggestedEnt2Score[id] = score_dicts[kkk][id]*weights[kkk]
            else:
                suggestedEnt2Score[id] += score_dicts[kkk][id]*weights[kkk]
        suggestedEnts.append(suggested[:b])
    return suggestedEnt2Score, suggestedEnts

def cmab(lefts, score_dicts, train_mapping, trained, ouf):
    t_total = time.time()
    N = len(score_dicts) # num of strategies
    R = 40 # num of rounds
    b = 50
    U = np.ones((N, R)) # for each arm, record its u
    num_chosen = np.ones(N)
    all_chosen = []

    d.ill_train_idx = copy.deepcopy(trained)
    d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
    experiment = Experiment(args=args)
    experiment.vali = True
    _, H1 = experiment.train_and_eval_val()
    H1_pre = H1*100
    # # in each iteration, each arm suggest b ents
    suggestedEnt2Score, suggestedEnts = suggesting_score(lefts, score_dicts, b, U, 1, num_chosen)
    selected = sorted(suggestedEnt2Score.items(), key=lambda d: d[1], reverse=True)[:b]
    chosen_ents = [i[0] for i in selected]
    all_chosen.extend(chosen_ents)
    lefts = list(set(lefts) - set(chosen_ents))

    #  update U for each arm, chose the overlapping part, and calculate the reward...
    # overlapping_total = [[],[],[]]
    num_chosen_this = np.zeros(N)
    for i in range(N):
        suggested = suggestedEnts[i]
        overlapping = list(set(suggested) & set(chosen_ents))
        num_chosen[i] += len(overlapping)
        num_chosen_this[i] = len(overlapping)
        new_train = []
        for item in overlapping:
            new_train.append([item, train_mapping[item]])
        if len(new_train)>0:
            d.ill_train_idx = copy.deepcopy(np.concatenate([trained, np.array(new_train)]))
            d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
            experiment = Experiment(args=args)
            experiment.vali = True
            _, H1 = experiment.train_and_eval_val()
            H1 = H1 * 100
            gap = H1 - H1_pre
            if gap < 0:
                gap = 0.0
        else:
            gap = 0.0
        U[i][0] = gap
    # print(num_chosen)

    # remove the selected from the score_dicts
    for i in range(N):
        for ent in chosen_ents:
            del score_dicts[i][ent]
        # print(len(score_dicts[i]))

    new_train = []
    for item in chosen_ents:
        new_train.append([item, train_mapping[item]])
    trained = np.concatenate([trained, np.array(new_train)])

    d.ill_train_idx = copy.deepcopy(trained)
    d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
    experiment = Experiment(args=args)
    experiment.vali = True
    _, H1 = experiment.train_and_eval_val()
    H1 = H1 * 100
    gap = H1 - H1_pre
    if gap < 0:
        gap = 0.0
    # print("chosen gain: " +str(gap))

    num_chosen_this = num_chosen_this/50.0
    for i in range(N):
        U[i][0] += gap*num_chosen_this[i]

    # print(len(set(all_chosen)))
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    for r in range(2,R+1):
        H1_pre = H1
        suggestedEnt2Score, suggestedEnts = suggesting_score(lefts, score_dicts, b, U, r, num_chosen)
        selected = sorted(suggestedEnt2Score.items(), key=lambda d: d[1], reverse=True)[:b]
        chosen_ents = [i[0] for i in selected]
        all_chosen.extend(chosen_ents)
        lefts = list(set(lefts) - set(chosen_ents))

        num_chosen_this = np.zeros(N)
        for i in range(N):
            suggested = suggestedEnts[i]
            overlapping = list(set(suggested) & set(chosen_ents))
            num_chosen[i] += len(overlapping)
            num_chosen_this[i] = len(overlapping)

            new_train = []
            for item in overlapping:
                new_train.append([item, train_mapping[item]])

            if len(new_train) > 0:
                d.ill_train_idx = copy.deepcopy(np.concatenate([trained, np.array(new_train)]))
                d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
                experiment = Experiment(args=args)
                experiment.vali = True
                _, H1 = experiment.train_and_eval_val()
                H1 = H1 * 100
                gap = H1 - H1_pre
                if gap < 0:
                    gap = 0.0
            else:
                gap = 0.0
            U[i][r-1] = gap

        # print(num_chosen)
        # remove the selected from the score_dicts
        for i in range(N):
            for ent in chosen_ents:
                del score_dicts[i][ent]

        new_train = []
        for item in chosen_ents:
            new_train.append([item, train_mapping[item]])
        trained = np.concatenate([trained, np.array(new_train)])

        d.ill_train_idx = copy.deepcopy(trained)
        d.ill_test_idx = copy.deepcopy(d.ill_val_idx)
        experiment = Experiment(args=args)
        experiment.vali = True
        _, H1 = experiment.train_and_eval_val()
        H1 = H1 * 100
        gap = H1 - H1_pre
        if gap < 0:
            gap = 0.0
        # print("chosen gain: " + str(gap))
        num_chosen_this = num_chosen_this / 50.0

        for i in range(N):
            U[i][r-1] += gap * num_chosen_this[i]

        if r%5==0:
            d.ill_train_idx = copy.deepcopy(trained)
            d.ill_test_idx = copy.deepcopy(d.ill_test_idx_)
            # print("Len of training " + str(len(d.ill_train_idx)))
            experiment = Experiment(args=args)
            enh_emb, HHHH = experiment.train_and_eval()
            ouf.write(str(HHHH) + '\n')
            ouf.flush()
            # update some dicts
            score_dicts[0] = update_dic_perc(lefts, score_dicts[0])
            score_dicts[1] = update_dic_perc(lefts, score_dicts[1])
            left2score_i = information_den(enh_emb, lefts)
            score_dicts[2] = left2score_i
            assert len(score_dicts[1]) == len(score_dicts[2]) == len(score_dicts[0])

        print("Already selecting " + str(len(set(all_chosen))) + " entities...")
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
        print()
    return all_chosen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epoch", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False,
                        help="whether to use early stop")  # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")

    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--encoder1", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100",
                        help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")
    parser.add_argument("--feat_adj_dropout", type=float, default=0.2, help="feat_adj_dropout")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1",
                        help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")

    parser.add_argument("--train_dist", type=str, default="euclidean",
                        help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean",
                        help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    parser.add_argument("--theta0", type=float, default=0.2, help="thres")  # 0.2
    parser.add_argument("--eta", type=float, default=0.003, help="thres")  # 0.003

    switch = 'GCNAPP_contras_active'  # APPtry1_comb_contras
    if switch == 'GCNAPP_contras_active':
        tw = 1;cf = 1;tr = False;tae = 1

    parser.add_argument("--model_name", type=str, default=switch,
                        help="name of the model, GCN, GCN_active, GCNAPP, GCNAPP_active, GCNAPP_contras, GCNAPP_contras_active")
    parser.add_argument("--two_views", type=int, default=tw,
                        help="whether to use two views, if not (0), the contra flag is also 0")
    parser.add_argument("--contras_flag", type=int, default=cf, help="whether to use contrastive learning")
    parser.add_argument("--train_random", action="store_true", default=tr,
                        help="random strategy for active training")  # !!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument("--train_add_embed", type=int, default=tae, help="whether to use embedding in active learning")

    parser.add_argument("--alp", type=float, default=0.2, help="balance two views, the first is GCN")  # 0.2
    parser.add_argument("--fuse_embed", type=int, default=1, help="fuse at the embedding level?")
    parser.add_argument("--appkk", type=int, default=5, help="fuse at the embedding level?")

    args = parser.parse_args()
    print(args)
    ouf = open('results/' + args.data_dir.split('/')[-1] + '_' + args.model_name + '_' + str(args.seed) + '.txt',
               'w')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    d = AlignmentData(data_dir=args.data_dir, rate=args.rate, share=args.share, swap=args.swap, val=args.val,
                      with_r=args.encoder.lower() == "naea")
    print(d)
    seed_num = 500
    seeds = d.ill_train_idx[:seed_num]
    train_active = copy.deepcopy(d.ill_train_idx[seed_num:])
    d.ill_train_idx = seeds
    trained = copy.deepcopy(seeds)

    # first round using 200 samples
    experiment = Experiment(args=args)
    t_total = time.time()
    enh_emb, H1 = experiment.train_and_eval()
    ouf.write(str(H1) + '\n')
    ouf.flush()
    print("optimization finished!")
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    train_mapping = dict()
    lefts = []
    rights = []
    for item in train_active:
        train_mapping[item[0]] = item[1]
        lefts.append(item[0])
        rights.append(item[1])

    left2score_dgeree = centrality_score(lefts, '_degree')
    left2score_pr = centrality_score(lefts, '_pr')
    left2score_i = information_den(enh_emb, lefts)

    chosen = cmab(lefts, [left2score_dgeree, left2score_pr, left2score_i], train_mapping, trained, ouf)
    print()