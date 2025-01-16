#!/bin/bash

# Export the current date and time for job labeling
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="emotion_recognition"
export JOB_NAME=train_CNN_"$LABEL"_"$DATE"

# Environment variables to optimize performance
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1

# Create directories for logs and scratch data
mkdir -p /scratch/${USER}/logs/emotion_recognition
mkdir -p /scratch/${USER}/data/emotion_recognition

# Submit the job
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME                                                    # Job name
#SBATCH -o /scratch/${USER}/logs/emotion_recognition/$JOB_NAME.%j.o     # Standard output log
#SBATCH -e /scratch/${USER}/logs/emotion_recognition/$JOB_NAME.%j.e     # Standard error log
#SBATCH --partition=gpu                                                 # GPU partition
#SBATCH --gres=gpu:1                                                    # Request 1 GPU
#SBATCH --ntasks=1                                                      # Number of tasks
#SBATCH --cpus-per-task=4                                               # Number of CPU cores per task
#SBATCH --mem=16G                                                       # Memory per node
#SBATCH --time=08:00:00                                                 # Max runtime
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu                      # Email for notifications
#SBATCH --mail-type=ALL                                                 # Notifications: BEGIN, END, FAIL

# Load necessary modules
module load cuda/11.7                                                   # Adjust to the appropriate CUDA version
module load pytorch/2.5.1                                               # Adjust to the appropriate PyTorch version

set -x

# Navigate to your project directory
cd $HOME/Emotion-Recognition                                            # Project directory

# Run your Python training script
srun python recognition_model.py --batch_size=32 --lr=0.0001 --epochs=10 --data_dir=/scratch/${USER}/data/CNN
EOT

# Submit job:
# sbatch amarel_jobscript.sh