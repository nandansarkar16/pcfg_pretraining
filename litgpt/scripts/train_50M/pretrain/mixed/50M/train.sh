#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:3              # Number of GPUs
#SBATCH --mem=64G                  # memory requested

litgpt pretrain --config ./custom_model_98_8M.yaml --data TextFiles --data.train_data_path ./train --data.val_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/val
EOT

