# ViLID (Anonymous Repo for Peer Review)

This repository contains the code for the paper **"ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation"**. ViLID is a novel framework for detecting nuanced misinformation by identifying fine-grained semantic misalignments between text and images. The system leverages AI-generated rationales to perform a human-like reasoning process, augmented by a principled cross-modal inconsistency score and a novel alignment regularization objective to enhance robustness and performance.


<img width="6214" height="3382" alt="ViLID_Pipeline" src="https://github.com/user-attachments/assets/e87b916e-c028-43fc-897f-deff57b01f5c" />


## Core Idea


The core of ViLID lies in its dual-pathway alignment analysis, which allows the system to:
* Quantify inconsistency at both the raw feature level ($S_{inc}$) and the higher-order semantic level ($S_r$) using AI-generated rationales.
* Integrate explicit reasoning into the detection process by comparing explanations of what the text claims versus what the image depicts.
* Utilize a novel alignment regularization loss ($L_{align}$) to enforce semantic consistency for truthful content, guiding the model to learn more robust and generalizable representations.
* Achieve state-of-the-art performance on challenging multimodal misinformation benchmarks.

## Repository Structure

The main components of this repository are organized as follows:

* `main.py`: The primary script to run training and evaluation. It handles data loading, model initialization, training loops, and final evaluation.
* `model.py`: Defines the core neural network architectures, including:
    * `TextEncoder` & `ImageEncoder`: Encapsulate the CLIP-based encoders for processing text and images.
    * `CrossModalAttention`: Implements the fusion transformer that integrates embeddings from text, images, and their corresponding rationales.
    * `ViLID`: The main model class that combines all components, calculates inconsistency scores, and makes final predictions.
    * `ViLIDLoss`: Implements the joint loss function, including the Binary Cross-Entropy loss and the custom alignment regularization term.
* `data_utils.py`: Contains utilities for dataset handling, including loading data from JSON files and managing image caching.
* `train.sh`: Slurm script for training with parameters and configs.
* `run_train.sh`: Slurm script to submit all training jobs.
* `test.sh`: Slurm script for testing with parameters and configs.
* `run_test.sh`: Slurm script to submit all testing jobs.

## Setup and Installation

1.  **Clone the repository.**
2.  **Python Environment:** A Python environment (e.g., Conda or venv) is recommended. The code is developed using Python 3.9 or later.
3.  **Install Dependencies:** Install the required Python packages from `ViLID.yml`. Key dependencies include:
    * `torch` (PyTorch)
    * `transformers` (Hugging Face Transformers)
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `Pillow` (PIL)

    You can typically install these using pip:
    ```bash
    conda env create -f ViLID.yml
    ```
    *(Note: Ensure PyTorch is installed according to your CUDA version if GPU support is needed.)*

## Running the Experiments

1.  **Prepare the Datasets:**
    * The experiments are designed for the **Fakeddit** and **MMFakeBench** datasets. For generalization, **M3A** is used.
    * Download the datasets from their original sources and place them in your data directory.

2.  **Generate Rationales:**
    * The model requires pre-generated rationales. You can use the provided prompts and HF models to generate rationales.

3.  **Configure Your Experiment:**
    * Modify the `train.sh` and `test.sh` files.

4.  **Run the Main Script:**
    Execute the `main.py` and `test.py` scripts.
    ```bash
    # To train a model
    python3 your path/ViLID/main.py

    # To evaluate a trained model
    python3 your path/ViLID/test.py
    ```

## Output

The script generates several outputs, saved in the directory specified by `output_dir` in the configuration file:

* **Log files (`.log`):** Detailed logs of the training and evaluation process.
* **Configuration file (`config.json`):** A copy of the exact configuration used for the run.
* **Model Checkpoints (`.pth`):** Saved model weights from the best-performing epoch (based on validation F1-score).
* **Evaluation Results (`test_results.json`):** A JSON file containing the final metrics (Accuracy, Precision, Recall, F1-Score) on the test set.

## Notes for Reviewers
This repository is provided for anonymous peer review. The code implements the ViLID framework and the experimental setup described in the submitted paper. The provided configuration files enable the reproduction of the reported experiments.

