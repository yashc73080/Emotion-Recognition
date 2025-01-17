#!/bin/bash

# Export the current date and time for job labeling
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="emotion_recognition"
export JOB_NAME=train_CNN_"$LABEL"_"$DATE"

# Environment variables to optimize performance
export OMP_NUM_THREADS=4 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1

# Create directories for logs and scratch data
mkdir -p /scratch/${USER}/logs/emotion_recognition
mkdir -p /scratch/${USER}/data/emotion_recognition

# Submit the job
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/emotion_recognition/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/emotion_recognition/$JOB_NAME.%j.e
#SBATCH --requeue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

module purge
module use /projects/community/modulefiles
module load cuda/11.7

set -x

cd $HOME/Emotion-Recognition/backend/model/

eval "$(conda shell.bash hook)"
conda activate emotion

python recognition_model.py --batch_size=32 --lr=0.0001 --epochs=10 --data_dir=/scratch/${USER}/Emotion-Recognition/dataset
EOT

# Submit job:
# sbatch amarel_jobscript.sh

# See logs:
# cd /scratch/$USER/logs/emotion_recognition