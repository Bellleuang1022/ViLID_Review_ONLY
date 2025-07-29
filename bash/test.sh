#!/bin/bash

#SBATCH --job-name=ViLID_test
#SBATCH --output=your path to output
#SBATCH --mail-user=your email
#SBATCH --mail-type=ALL
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=your partition
#SBATCH --gres=gpu:1
#SBATCH --qos=your server group

echo "Start Date : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Job Name   : $SLURM_JOB_NAME"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"
echo "SLURM CPUs on node: $SLURM_CPUS_ON_NODE"
echo "SLURM GPUs on node: $SLURM_GPUS_ON_NODE"
echo "SLURM CPU cores requested: $SLURM_CPUS_PER_TASK"
echo "--------------------------------------------------------------------"
start_time=$(date +%s)

# Load modules & environment
module purge
module load conda cuda
echo "Modules loaded:"
module list
echo "--------------------------------------------------------------------"

# Activate Conda environment
export PATH=your environment path/bin:$PATH
source activate your environment path 
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available in PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "--------------------------------------------------------------------"

# Hugging Face cache & token
export HF_HOME="your path for HF cache"
export TORCH_HOME="your path for torch cache"
export HF_ACCESS_TOKEN="your token for HF"
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

if [ -z "$HF_ACCESS_TOKEN" ]; then
    echo "Error: HF_ACCESS_TOKEN is not set."
    exit 1
fi
echo "HF_HOME set to: $HF_HOME"
echo "--------------------------------------------------------------------"
echo "Starting ViLID testing"
echo "--------------------------------------------------------------------"

# Positional arguments
MODEL_PATH=$1    
TEST_DATA=$2       
OUTPUT_DIR=$3      

# Ensure output directory exists (test.py also does this, but good practice)
mkdir -p "$OUTPUT_DIR"

# Define the path to your test.py script
TEST_SCRIPT_PATH="your path/ViLID/test.py"

echo "Model Path: $MODEL_PATH"
echo "Test Data: $TEST_DATA"
echo "Output Dir: $OUTPUT_DIR"
echo "Test script: $TEST_SCRIPT_PATH"
echo "--------------------------------------------------------------------"

# Run the test script
python3 "$TEST_SCRIPT_PATH" \
    --model_path        "$MODEL_PATH" \
    --test_data         "$TEST_DATA" \
    --output_dir        "$OUTPUT_DIR" \
    --batch_size        64 \
    --gpu_id            "0" \
    --use_amp \
    --threshold         0.5 \
    --save_preds \
    --num_workers       4 \
    --use_image_cache \
    --image_cache_dir   "your image cache directory" \
    --image_cache_size_gb 20.0 \
    --config_path       "your config file path" \

# Capture exit status of the python script
status=$?
if [ $status -ne 0 ]; then
    echo "Error: Python script failed with status $status"
    # You might want to exit here or let the script continue to log time
fi
echo "--------------------------------------------------------------------"
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

days=$(( elapsed/86400 ))
hours=$(( (elapsed%86400)/3600 ))
mins=$(( (elapsed%3600)/60 ))
secs=$(( elapsed%60 ))

echo
echo "Python script execution finished with status: $status"
echo "Elapsed time: ${days}d ${hours}h ${mins}m ${secs}s"
echo "End Date   : $(date)"
echo "Host       : $(hostname -s)"
echo "Directory  : $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID finished."