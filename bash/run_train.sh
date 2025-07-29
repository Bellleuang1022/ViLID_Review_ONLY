#!/bin/bash

# List of data files
data_files=(
    "your path to data files here"
)

# List of random seeds
seeds=(42 128 1008 3407)

# Log directory
log_dir="your log directory"
mkdir -p "$log_dir"

for data_file in "${data_files[@]}"; do
    # Extract dataset name (e.g., Fakeddit, MMFakeBench)
    if [[ "$data_file" == *"Fakeddit"* ]]; then
        dataset="Fakeddit"
    elif [[ "$data_file" == *"MMFakeBench"* ]]; then
        dataset="MMFakeBench"
    else
        echo "Unknown dataset in: $data_file"
        continue
    fi

    # Extract filename
    file_name=$(basename "$data_file")
    data_name="${dataset}_${file_name}"

    for seed in "${seeds[@]}"; do
        job_name="vilid_${dataset}_${file_name%.json}_seed${seed}"
        log_path="${log_dir}/${job_name}.log"

        echo "Submitting job: $job_name"
        sbatch --job-name="$job_name" \
               --output="$log_path" \
               your path to train.sh \
               "$data_file" "$data_name" "$seed"
    done
done
