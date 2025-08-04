#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:4             # Number of GPUs


litgpt pretrain --config ./custom_model_98_8M.yaml --data TextFiles --data.train_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/train_without_trees --data.val_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/val
EOT

