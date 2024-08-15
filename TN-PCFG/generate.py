import torch
import argparse
from parser.helper.data_module import DataModule
from parser.helper.util import *
from easydict import EasyDict as edict
import yaml
from transformers import GPT2Tokenizer


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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model.load_state_dict(torch.load('/data/cl/u/nsarkar/pcfg_pretrain/TN-PCFG/log/simple_npcfg_nt4096_t8192_curriculum0/SNPCFG2024-07-22-18_42_38/best.pt', map_location=args.device))
log.info(f"Loaded trained weights")

data_path = '/data/cl/u/nsarkar/generated_data/pcfg/generated_data.txt'
tree_data_path = '/data/cl/u/nsarkar/generated_data/pcfg/generated_data_tree.txt'
if os.path.exists(data_path):
    with open(data_path, 'r', errors='ignore') as f:
        text = f.read()
        total_tokens_generated_so_far = len(tokenizer.tokenize(text))
else:
    total_tokens_generated_so_far = 0


with open(data_path, 'a') as f: #open(tree_data_path, 'a') as f_tree:
    while total_tokens_generated_so_far < 1000000000:
        result = model.generate()
        if result is None:
            continue
        vocab_ids, string_tree_rep = result
        tokens = [dataset.word_vocab.to_word(vocab_id) for vocab_id in vocab_ids]
        text = ' '.join(tokens) + '\n'
        tree_string = [
            (dataset.word_vocab.to_word(word_id) if word_id >= 0 
            else ('ı' if word_id == -1 else '§'))
            for word_id in string_tree_rep
        ]
        tree_string = ' '.join(tree_string) + '\n'

        full_text = text + tree_string + '\n'
        total_tokens_generated_so_far += len(tokenizer.tokenize(text))
        log.info(f"Total tokens generated so far: {total_tokens_generated_so_far}")
        f.write(full_text)
        f.flush()
        #f_tree.write(tree_string)

