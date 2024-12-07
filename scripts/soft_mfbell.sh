#!/bin/bash
#SBATCH --job-name=256sfbell
#SBATCH --output=/lustre/scratch/client/vinai/users/phinh2/workspace/softmoe/result/soft_fbell.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/phinh2/workspace/softmoe/result/soft_fbell_err.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=sdc2-hpc-dgx-a100-015
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=24
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
conda activate deit
cd /lustre/scratch/client/vinai/users/phinh2/workspace/softmoe
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

echo "Training ..."
python -m torch.distributed.launch --master_port=2712 --nproc_per_node=2 --use_env main3.py --model soft_moe_vit_tiny --batch-size 256 \
 --data-path /lustre/scratch/client/vinai/users/phinh2/workspace/dataset/imagenet --output_dir /lustre/scratch/client/vinai/users/phinh2/workspace/softmoe/result/soft_2fbell

