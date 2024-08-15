#!/bin/bash


sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT

sbatch <<EOT
#!/bin/bash
#SBATCH --partition=long       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --requeue

python generate.py --conf /data/cl/u/nsarkar/TN-PCFG/config/simplepcfg/simple_npcfg_nt4096_t8192_curriculum0.yaml
EOT
