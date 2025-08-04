#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs


litgpt pretrain --config ./custom_model_98_8M.yaml --data TextFiles --data.train_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/train_without_trees/ --data.val_data_path ~/pcfg_pretrain/Wikipedia/fifty_million/val
EOT


# Parsing trees from real data using baseline pcfg script here - will create a separate script for this later
# sbatch <<EOT
# #!/bin/bash
# #SBATCH --partition=standard       # Partition name
# #SBATCH --gres=gpu:1               # Number of GPUs


# python evaluate.py --load_from_dir /data/cl/u/nsarkar/pcfg_pretrain/TN-PCFG/log/simple_npcfg_nt4096_t8192_curriculum0/SNPCFG2024-12-02-14_19_42 --create_tree True
# EOT