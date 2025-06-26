import torch
import argparse
from parser.helper.data_module import DataModule
from parser.helper.util import *
from easydict import EasyDict as edict
import yaml
from transformers import GPT2TokenizerFast
import pickle


parser = argparse.ArgumentParser(
        description='PCFGs'
    )
parser.add_argument('--conf', '-c', default='')
args2 = parser.parse_args()
yaml_cfg = yaml.load(open(args2.conf, 'r'), Loader=yaml.Loader)
args = edict(yaml_cfg)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
log = get_logger(args)


dataset = DataModule(args)
log.info("Created the Dataset")
model = get_model(args.model, dataset)
log.info("Created the Model")

tokenizer = GPT2TokenizerFast.from_pretrained('/data/cl/u/nsarkar/pcfg_pretrain/litgpt/checkpoints/custom_tokenizer_30k_vocab/saved_tokenizer')
log.info("Loaded the tokenizer")

model.load_state_dict(torch.load('/data/cl/u/nsarkar/pcfg_pretrain/TN-PCFG/log/simple_npcfg_nt4096_t8192_curriculum0/SNPCFG2024-12-02-14_19_42/best.pt', map_location=args.device)) # 30k vocab size, PCFG trained on 50M tokens
log.info(f"Loaded trained weights")


total_tokens_generated_so_far = 0

# file_ind = 0
word_data_path = f"/data/cl/u/nsarkar/generated_data/pcfg/randomly_initialized/test/text/pretrain_words.txt"
tree_data_path = f"/data/cl/u/nsarkar/generated_data/pcfg/randomly_initialized/test/text/pretrain_trees.txt"

while total_tokens_generated_so_far < 10000000:
    result = model.generate()
    if result is None:
        continue
    vocab_ids, string_tree_rep = result

    words = [dataset.word_vocab.to_word(vocab_id) for vocab_id in vocab_ids]
    text = ' '.join(words) + '\n'
    tree_string = [
        (dataset.word_vocab.to_word(word_id) if word_id >= 0 
        else ('ı' if word_id == -1 else '§'))
        for word_id in string_tree_rep
    ]
    tree_string = ' '.join(tree_string) + '\n'

    tokens_generated = len(tokenizer.tokenize(text))
    total_tokens_generated_so_far += tokens_generated
    with open(word_data_path, 'a') as f:
        f.write(text) 
    with open(tree_data_path, 'a') as f:
        f.write(tree_string)

    log.info(f"Total tokens generated so far: {total_tokens_generated_so_far}\n")


