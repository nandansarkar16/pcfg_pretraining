#!/bin/bash



sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_pcfg_10_on_wiki.yaml --data TextFiles --data.train_data_path ~/Wikipedia/10M/train/ --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_pcfg_25_on_wiki.yaml --data TextFiles --data.train_data_path ~/Wikipedia/10M/train/ --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_pcfg_50_on_wiki.yaml --data TextFiles --data.train_data_path ~/Wikipedia/10M/train/ --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_pcfg_100_on_wiki.yaml --data TextFiles --data.train_data_path ~/Wikipedia/10M/train/ --data.val_data_path ~/Wikipedia/10M/val/
EOT
