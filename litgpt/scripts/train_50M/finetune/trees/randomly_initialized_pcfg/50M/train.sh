#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --mem=64G
#SBATCH --exclude=doodle,samoyed,terrier,mastiff


litgpt pretrain --config ./custom_model_98_8M.yaml --data TextFiles --data.train_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/train_without_trees --data.val_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/val
EOT

