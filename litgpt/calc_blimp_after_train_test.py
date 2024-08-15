from litgpt.model import GPT
from litgpt.config import Config
import torch
from litgpt.calculate_blimp_scores import eval_all_blimp

config = Config(
    n_embd=512,
    n_head=8,
    n_layer=12,
    padded_vocab_size=50304, # gpt2 tokenizer has vocab of 50257 so padded to 64/128 is 50304
    block_size=1024,
    norm_class_name="RMSNorm", # based off micro_LLaMA
    mlp_class_name="LLaMAMLP", # based off micro_LLaMA
    norm_eps=1e-5, # based off micro_LLaMA
    rotary_percentage=1.0, # based off micro_LLaMA
    parallel_residual=True, # False is what micro_LLaMA does
    bias=True, # False is what micro_LLaMA does
    n_query_groups=4,
    intermediate_size=2048
)

model = GPT(config).to('cuda')
print("Model created!")

paths = [
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_10_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_25_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_50_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_100_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_10_tree_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_25_tree_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_50_tree_on_wiki/best/lit_model.pth',
    '/data/cl/u/nsarkar/litgpt/scripts/train/98_8M/wiki/run_1/out/custom_model_98_8M_pcfg_100_tree_on_wiki/best/lit_model.pth'
]

states_list = [torch.load(path, map_location='cuda')['model'] for path in paths]
print("States loaded!")


averages_list = []
for i, state_dict in enumerate(states_list): 
    total_score = 0 
    model.load_state_dict(state_dict)
    print(f"On model {i}!")
    for j in range(4):
        score = eval_all_blimp(model, model.max_seq_length)['Average']
        total_score += score
        print(f"\tFinished BLiMP {j}!")
    average = total_score / 4
    averages_list.append(average)

print(averages_list)
