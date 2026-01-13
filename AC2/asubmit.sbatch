#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=run.out
#SBATCH --error=run.out
#SBATCH --nodes=1
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_109_tandon_advanced
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --wrap "hostname && sleep infinity"
#SBATCH --gres=gpu:1

ulimit -l unlimited
ulimit -s unlimited

export OMP_NUM_THREADS=1

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/aq2212/environments/pysages
python_exec=/scratch/aq2212/environments/pysages/bin/python
${python_exec} -u train.py
${python_exec} -u eval.py
