#!/bin/bash -l
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#SBATCH --mem=16g
#SBATCH --tmp=16g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=liang656@umn.edu 

#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name="test_tree_ring"
#SBATCH --output=test_tree_ring.out
#SBATCH --error=test_tree_ring.err

cd /scratch.global/liang656/tree-ring-watermark/
export PATH=/home/jusun/liang656/miniconda3/envs/tree-ring/bin:$PATH

python collect_tree_ring_dataset.py --dataset coco