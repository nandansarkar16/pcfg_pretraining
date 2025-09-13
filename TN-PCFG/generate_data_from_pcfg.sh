#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

python generate.py --conf /data/cl/u/nsarkar/pcfg_pretrain/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT
