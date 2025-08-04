#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ./custom_model_98_8M.yaml --data MatrixTextFiles --data.train_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/train_with_trees/temperature_1_2 --data.val_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/val
EOT

