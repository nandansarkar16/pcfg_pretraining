#!/bin/bash



sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_on_pcfg_10.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/string/10M --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:2              # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_on_pcfg_25.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/string/25M --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:2               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_on_pcfg_50.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/string/50M --data.val_data_path ~/Wikipedia/10M/val/
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:3               # Number of GPUs


litgpt pretrain --config ~/pcfg_pretrain/litgpt/config_hub/pretrain/98_8M/wiki/custom_model_98_8M_on_pcfg_100.yaml --data TextFiles --data.train_data_path ~/generated_data/pcfg/string/100M --data.val_data_path ~/Wikipedia/10M/val/
EOT