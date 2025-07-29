#!/bin/bash

echo "Starting batch test submission process..."
echo "Date: $(date)"
echo "--------------------------------------------------------------------"

# --- Configuration ---
# Base directory where your trained model .pt files are stored
MODEL_FILES_DIR=" "

# Base directory where your test split .csv files are stored
TEST_DATA_SPLITS_DIR=" "

# Base directory where the output of each test.py run (metrics.json, plots, etc.) will be saved.
# A subdirectory will be created here for each individual test run.
RESULTS_OUTPUT_BASE_DIR=" "

# Directory to store the SLURM log files (stdout/stderr) for each sbatch job submitted by this script.
SBATCH_LOGS_DIR=" "

# Full path to your test.sh wrapper script
TEST_WRAPPER_SCRIPT_PATH=" " 

# --- Sanity Checks & Setup ---
if [ ! -d "$MODEL_FILES_DIR" ]; then
    echo "ERROR: Model files directory not found: $MODEL_FILES_DIR"
    exit 1
fi
if [ ! -d "$TEST_DATA_SPLITS_DIR" ]; then
    echo "ERROR: Test data splits directory not found: $TEST_DATA_SPLITS_DIR"
    exit 1
fi
if [ ! -f "$TEST_WRAPPER_SCRIPT_PATH" ]; then
    echo "ERROR: test.sh script not found: $TEST_WRAPPER_SCRIPT_PATH"
    exit 1
fi

mkdir -p "$RESULTS_OUTPUT_BASE_DIR"
mkdir -p "$SBATCH_LOGS_DIR"

# Find all .pt files in the model directory
find "$MODEL_FILES_DIR" -maxdepth 1 -name "*.pt" -print0 | while IFS= read -r -d $'\0' model_file_full_path; do
    model_filename=$(basename "$model_file_full_path")

    echo "Processing model file: $model_filename"

    # Parse model_filename to extract components.
    # Expected format: seed<SEED>_data<DATASET>_<MODEL_VARIANT>_rationales.json_<TIMESTAMP>.pt
    if [[ "$model_filename" =~ ^seed([0-9]+)_data([a-zA-Z0-9]+)_([a-zA-Z0-9._-]+)_rationales\.json_([0-9]+)\.pt$ ]]; then
        seed="${BASH_REMATCH[1]}"
        dataset="${BASH_REMATCH[2]}"       
        model_variant="${BASH_REMATCH[3]}" 
        timestamp="${BASH_REMATCH[4]}"     

        echo "  Parsed components: Seed=$seed, Dataset=$dataset, ModelVariant=$model_variant, Timestamp=$timestamp"

        # Construct the corresponding test data CSV filename
        # Expected format: test_<DATASET>_<MODEL_VARIANT>_rationales.json_<MODEL_VARIANT>_rationales_seed<SEED>.csv
        test_data_filename="test_${dataset}_${model_variant}_rationales.json_${model_variant}_rationales_seed${seed}.csv"
        test_data_full_path="${TEST_DATA_SPLITS_DIR}/${test_data_filename}"

        # Check if the constructed test data file exists
        if [ -f "$test_data_full_path" ]; then
            echo "  Found corresponding test data file: $test_data_filename"

            # Define the specific output directory for this test.sh run's results
            # This directory will be passed as the third argument to test.sh
            # and test.sh will instruct test.py to save its outputs there.
            current_test_output_dir="${RESULTS_OUTPUT_BASE_DIR}/${dataset}_${model_variant}_seed${seed}_eval_run"
            mkdir -p "$current_test_output_dir" 

            # Define SLURM job name and log file path for this specific sbatch submission
            sbatch_job_name="test_${dataset}_${model_variant}_s${seed}"
            sbatch_log_file="${SBATCH_LOGS_DIR}/${sbatch_job_name}.log" 

            echo "  Output directory for test.py results: $current_test_output_dir"
            echo "  SLURM Job Name: $sbatch_job_name"
            echo "  SLURM Log File: $sbatch_log_file"
            echo "  Submitting sbatch job..."

            # Submit the job using sbatch, passing the required arguments to test.sh
            sbatch --job-name="$sbatch_job_name" \
                   --output="$sbatch_log_file" \
                   "$TEST_WRAPPER_SCRIPT_PATH" \
                   "$model_file_full_path" \
                   "$test_data_full_path" \
                   "$current_test_output_dir"

            echo "  sbatch submission command executed for $model_filename."
            echo "--------------------------------------------------------------------"
        else
            echo "  WARNING: Test data file NOT FOUND: $test_data_full_path"
            echo "  Skipping sbatch submission for model: $model_filename"
            echo "--------------------------------------------------------------------"
        fi
    else
        echo "  WARNING: Could not parse model filename according to expected pattern: $model_filename"
        echo "  Skipping this file."
        echo "--------------------------------------------------------------------"
    fi
done

