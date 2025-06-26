# -*- coding: utf-8 -*-

from parser.cmds.cmd import CMD
import torch

from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.loader_wrapper import DataPrefetcher
import torch
import numpy as np
from parser.helper.util import *
from parser.helper.data_module import DataModule
import click
import pickle

class Evaluate(CMD):

    def __call__(self, args, eval_dep=False, decode_type='mbr'):
        super(Evaluate, self).__call__(args)
        self.device = args.device
        self.args = args
        self.create_tree = args.create_tree
        dataset = DataModule(args)
        self.model = get_model(args.model, dataset)
        best_model_path = self.args.load_from_dir + "/best.pt"
        self.model.load_state_dict(torch.load(str(best_model_path)))
        print('successfully load')

        test_loader = dataset.test_dataloader
        test_loader_autodevice = DataPrefetcher(test_loader, device=self.device)
        if not eval_dep:
            likelihood, predicted_trees = self.evaluate(test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type)
        else:
            metric_uas, likelihood, predicted_trees = self.evaluate(test_loader_autodevice, eval_dep=eval_dep, decode_type=decode_type)
            print(metric_uas)
        #print(metric_f1)
        print("here")
        if self.create_tree:
            print("Creating trees")
            predicted_trees_list = []  
            word_total_in_file = 0
            file_ind = 0
            data_path = f"/data/cl/u/nsarkar/pcfg_pretrain/Wikipedia/fifty_million/train_with_trees/50_wiki_parsed.txt"
            # Get predicted trees, and convert vocab_ids to english words
            for batch_idx, batch in enumerate(predicted_trees):
                for tree_idx, tree in enumerate(batch):
                    tree_str = ' '.join(tree)
                    predicted_trees_list.append(tree_str)
                    #print(predicted_trees_list[-1], flush=True)
                    with open(data_path, 'a') as f:
                        f.write(tree_str + '\n')
                    # words = [word for word in tree if word not in ['ı', '§']]
                    # words_str = ' '.join(words)
                    # words_str += '\n'
                    # syntactic_distance_dict["words"].append(words_str)
                    # word_total_in_file += len(words)

                    # if word_total_in_file >= 100:
                    #     # Save data to file
                    #     with open(data_path, 'w') as f:
                    #         for tree in predicted_trees_list:
                    #             f.write(tree + '\n')
                    #     print(f"Saved data to {data_path}")
                    #     print("Words in file: ", word_total_in_file)
                    #     # Reset data
                    #     predicted_trees_list = []
                    #     word_total_in_file = 0
                    #     file_ind += 1
                    #     data_path = f"/data/cl/u/nsarkar/pcfg_pretrain/Wikipedia/fifty_million/train_with_trees/test/2_50_wiki_with_trees_{file_ind}.txt"


            # Final save    
            # with open(data_path, 'w') as f:
            #     for tree in predicted_trees_list:
            #         f.write(tree + '\n')
            # print(f"Saved data to {data_path}")
            # print("Words in file: ", word_total_in_file)

        print(likelihood)








