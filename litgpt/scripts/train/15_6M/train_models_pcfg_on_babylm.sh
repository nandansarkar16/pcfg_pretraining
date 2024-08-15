#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/15_6M/pcfg_gen_data/custom_model_15_6M_pcfg_10_on_babylm.yaml --data TextFiles --data.train_data_path ~/BabyLM_cleaned/train_set/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/15_6M/pcfg_gen_data/custom_model_15_6M_pcfg_25_on_babylm.yaml --data TextFiles --data.train_data_path ~/BabyLM_cleaned/train_set/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/15_6M/pcfg_gen_data/custom_model_15_6M_pcfg_50_on_babylm.yaml --data TextFiles --data.train_data_path ~/BabyLM_cleaned/train_set/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

litgpt pretrain --config /data/cl/u/nsarkar/litgpt/config_hub/pretrain/15_6M/pcfg_gen_data/custom_model_15_6M_pcfg_100_on_babylm.yaml --data TextFiles --data.train_data_path ~/BabyLM_cleaned/train_set/ --data.val_data_path ~/BabyLM_cleaned/val_set\/
EOT

