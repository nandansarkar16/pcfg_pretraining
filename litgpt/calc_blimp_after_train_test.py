from litgpt.model import GPT
from litgpt.config import Config
import torch
from litgpt.calculate_blimp_scores import eval_all_blimp
import pandas as pd

config = Config(
    n_embd=512,
    n_head=8,
    n_layer=16,
    padded_vocab_size=30016, 
    block_size=1024,
    norm_class_name="RMSNorm", 
    mlp_class_name="LLaMAMLP", 
    norm_eps=1e-5, 
    rotary_percentage=1.0, 
    parallel_residual=True, 
    bias=True,
    n_query_groups=4,
    intermediate_size=2048
)

model = GPT(config).to('cuda')
print("Model created!")

paths = [
    '/data/cl/u/nsarkar/pcfg_pretrain/litgpt/scripts/train_50M/finetune/mixed/0_5M/out/custom_model_98_8M_finetuned/best/lit_model.pth',
    '/data/cl/u/nsarkar/pcfg_pretrain/litgpt/scripts/train_50M/finetune/mixed/1M/out/custom_model_98_8M_finetuned/best/lit_model.pth',
    '/data/cl/u/nsarkar/pcfg_pretrain/litgpt/scripts/train_50M/finetune/mixed/10M/out/custom_model_98_8M_finetuned/best/lit_model.pth',
    '/data/cl/u/nsarkar/pcfg_pretrain/litgpt/scripts/train_50M/finetune/mixed/30M/out/custom_model_98_8M_finetuned/best/lit_model.pth',
    '/data/cl/u/nsarkar/pcfg_pretrain/litgpt/scripts/train_50M/finetune/mixed/50M/out/custom_model_98_8M_finetuned/best/lit_model.pth',
]

states_list = [torch.load(path, map_location='cuda')['model'] for path in paths]
print("States loaded!")


results = {}
for i, state_dict in enumerate(states_list):
    model.load_state_dict(state_dict)
    print(f"Evaluating model {i}...")
    
    avg_score = eval_all_blimp(model, model.max_seq_length)
    for task, score in avg_score.items():
        if task not in results:
            results[task] = {}
        results[task][f"model_{i}"] = score
        print(f"\t{task}: {score}")
    print(f"\tAvg Score: {score}")

df = pd.DataFrame(results).T  
df.index.name = "BLiMP Subtask"
df.reset_index(inplace=True)
output_path = "all_model_blimp_scores.csv"
df.to_csv(output_path, index=False)
print(f"Saved all results to {output_path}")

