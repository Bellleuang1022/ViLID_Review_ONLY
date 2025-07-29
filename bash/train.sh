#!/bin/bash

#SBATCH --mail-user=your email
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1                    # Single task, managed by torchrun
#SBATCH --cpus-per-task=5             # Total CPU cores for the task
#SBATCH --mem-per-cpu=10gb             # Job Memory
#SBATCH --partition=your partition
#SBATCH --gres=gpu:1
#SBATCH --qos=your server group

echo "Start Date : $(date)"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"
echo "Running $SLURM_JOB_NAME on $SLURM_CPUS_ON_NODE CPU cores and $SLURM_GPUS_ON_NODE GPUs"
echo "--------------------------------------------------------------------"
start_time=$(date +%s)

# Load necessary modules and environment
module purge
module load conda cuda
module list

export PATH=your environment path/bin:$PATH
conda activate your environment

# Hugging Face cache & token
export HF_HOME="your path for HF cache"
export TORCH_HOME="your path for torch cache"
export HF_ACCESS_TOKEN="your token for HF"
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

if [ -z "$HF_ACCESS_TOKEN" ]; then
    echo "Error: HF_ACCESS_TOKEN is not set."
    exit 1
fi

echo "--------------------------------------------------------------------"
echo "Starting ViLID training"
echo "--------------------------------------------------------------------"


DATA_FILE=$1
DATA_NAME=$2
SEED=$3

python3 your path/ViLID/main.py \
  --data_file "$DATA_FILE" \
  --data_name "$DATA_NAME" \
  --text_encoder "openai/clip-vit-base-patch32" \
  --image_encoder "openai/clip-vit-base-patch32" \
  --hidden_size 512 \
  --num_fusion_layers 4 \
  --num_fusion_heads 8 \
  --dropout 0.1 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-6 \
  --encoder_lr 1e-6 \
  --weight_decay 0.001 \
  --gamma 0.1 \
  --beta 5.0 \
  --early_stop 5 \
  --use_amp \
  --clip_grad 1.0 \
  --use_image_cache \
  --image_cache_dir "your image cache directory" \
  --image_cache_size_gb 2 \
  --num_workers 2 \
  --output_dir "your output directory" \
  --checkpoint_dir "your checkpoint directory" \
  --save_every 1 \
  --mode train \
  --seed "$SEED" \
  --use_lr_scheduler

echo "--------------------------------------------------------------------"
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

days=$(( elapsed/86400 ))
hours=$(( (elapsed%86400)/3600 ))
mins=$(( (elapsed%3600)/60 ))
secs=$(( elapsed%60 ))

echo
echo "Elapsed time: ${days}d ${hours}h ${mins}m ${secs}s"
echo "End Date : $(date)"
echo "Host     : $(hostname -s)"
echo "Directory: $(pwd)" 