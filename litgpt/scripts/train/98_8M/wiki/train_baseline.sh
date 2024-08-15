#!/bin/bash



sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M.yaml --data TextFiles --data.train_data_path ~/Wikipedia/10M/train/ --data.val_data_path ~/Wikipedia/10M/val/
EOT
