#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/75_1M/pcfg_gen_data/custom_model_75_1M_on_pcfg_10.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/10M_tokens/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/75_1M/pcfg_gen_data/custom_model_75_1M_on_pcfg_25.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/25M_tokens/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/75_1M/pcfg_gen_data/custom_model_75_1M_on_pcfg_50.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/50M_tokens/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/75_1M/pcfg_gen_data/custom_model_75_1M_on_pcfg_100.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/100M_tokens/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT

