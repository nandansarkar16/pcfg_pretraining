# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from parser.helper.metric import LikelihoodMetric,  UF1, LossMetric, UAS

import time

class CMD(object):
    def __call__(self, args):
        self.args = args

    def train(self, loader):
        self.model.train()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        train_arg = self.args.train
        for x, _ in t:

            self.optimizer.zero_grad()
            loss = self.model.loss(x)
            loss.backward()
            if train_arg.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                     train_arg.clip)
            self.optimizer.step()
            t.set_postfix(loss=loss.item())
        return

    def produce_tree(self, x, prediction):
        # x is a tensor of shape (batch_size, seq_len)
        # prediction is a list of lists of integers

        x = x['orginal_word']
        predicted_trees = []
        for batch_idx in range(len(x)):
            predicted_tree = list(map(str, x[batch_idx].copy()))  # Convert each element to a plain Python string
            for span in prediction[batch_idx]:
                start, end = span
                # skip the span if it is a single word
                if start + 1 == end:
                    continue
                # Find the corresponding start and end indices of the tree
                tree_start_ind, tree_end_ind, token_idx = 0, 0, 0
                for tree_idx in range(len(predicted_tree)):
                    if predicted_tree[tree_idx] not in ['ı', '§']:
                    # Decrease start and end counters when we pass non-bracket tokens
                        if token_idx == start:
                            tree_start_ind = tree_idx
                        if token_idx == end - 1:
                            tree_end_ind = tree_idx
                            break
                        token_idx += 1

                predicted_tree = predicted_tree[:tree_start_ind] +  ['ı'] + predicted_tree[tree_start_ind:tree_end_ind + 1] + ['§'] + predicted_tree[tree_end_ind + 1:]
            predicted_trees.append(predicted_tree)
        
        return predicted_trees



    @torch.no_grad()
    def evaluate(self, loader, eval_dep=False, decode_type='mbr', model=None):
        if model == None:
            model = self.model
        model.eval()
        #metric_f1 = UF1()
        if eval_dep:
            metric_uas = UAS()
        metric_ll = LikelihoodMetric()
        t = tqdm(loader, total=int(len(loader)),  position=0, leave=True)
        print('decoding mode:{}'.format(decode_type))
        print('evaluate_dep:{}'.format(eval_dep))
        predicted_trees = []
        for x, y in t:
            result = model.evaluate(x, decode_type=decode_type, eval_dep=eval_dep)
            #metric_f1(result['prediction'], y['gold_tree'])
            predicted_trees.append(self.produce_tree(x, result['prediction']))
            metric_ll(result['partition'], x['seq_len'])
            if eval_dep:
                metric_uas(result['prediction_arc'], y['head'])
        if not eval_dep:
            #return metric_f1, metric_ll
            #print("predicted trees: ", predicted_trees)
            return metric_ll, predicted_trees
        else:
            #return metric_f1, metric_uas, metric_ll
            return metric_uas, metric_ll, predicted_trees




