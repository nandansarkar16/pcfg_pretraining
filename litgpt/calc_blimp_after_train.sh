sbatch <<EOT
#!/bin/bash
#SBATCH --partition=standard       # Partition name
#SBATCH --gres=gpu:1               # Number of GPUs

python calc_blimp_after_train_test.py
EOT