ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation


This repository contains the official implementation for the paper: "ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation".

ViLID is a novel framework designed to identify fine-grained semantic misalignments between text and images in multimodal content. It addresses the challenge of detecting nuanced misinformation by integrating direct feature-level inconsistency with higher-order reasoning based on AI-generated rationales.

📜 Abstract
The proliferation of multimodal misinformation, characterized by the integration of text and images to create deceptive narratives, presents a significant societal challenge. To fill this gap, we introduce ViLID, a novel framework that excels at identifying fine-grained semantic misalignments between text and images. Key contributions include: a principled cross-modal inconsistency score; a rationale-augmented reasoning module utilizing LLMs to produce explicit textual and visual explanations; and a novel alignment regularization term to enhance model robustness. Extensive evaluations on the Fakeddit and MMFakeBench benchmarks demonstrate that ViLID achieves state-of-the-art performance in misinformation detection.

🏗️ Model Architecture
ViLID's architecture uses a dual-pathway analysis to detect inconsistencies. As shown in the diagram below, every text-image pair is processed through both a Non-Reasoning Pathway (for direct feature extraction) and a Reasoning Pathway (which uses LLMs to generate and encode rationales). The features and inconsistency scores from both pathways are then combined in a fusion transformer and passed to a classifier to make the final prediction.

Figure 1: High-level overview of the ViLID architecture.

✨ Key Features
Rationale-Augmented Reasoning: Integrates LLM-generated rationales to capture higher-order semantic inconsistencies between text and image explanations.

Dual Inconsistency Scores: Computes two separate scores: $S_{inc}$ for direct feature mismatch and $S_{r}$ for rationale-level mismatch, providing the model with explicit alignment signals.

Alignment Regularization: A novel loss term ($L_{align}$) encourages the model to learn consistent representations for truthful data, improving robustness and generalization.

State-of-the-Art Performance: Achieves SOTA performance on challenging benchmarks like Fakeddit and MMFakeBench.

⚙️ Setup and Installation
To get started, clone the repository and set up the Conda environment using the provided files.

# 1. Clone the repository
git clone https://github.com/your-username/ViLID.git
cd ViLID

# 2. Create and activate the Conda environment
conda create -n vilid python=3.9
conda activate vilid

# 3. Install dependencies
pip install -r requirements.txt

📊 Datasets
Our experiments use the following datasets. Please follow the instructions from the original sources to download them.

Fakeddit: A large-scale multimodal dataset from Reddit. You can find more information in the original paper by Nakamura et al. (2019). Due to its size, we use a 25% stratified sample for our experiments.

MMFakeBench: A modern benchmark designed with mixed-source misinformation. More details are available in the paper by Liu et al. (2024).

M3A: Used for our zero-shot cross-domain evaluation. See Xu et al. (2024) for access details.

Pre-generated Rationales
Our model relies on pre-generated rationales for training. We provide the rationales used in our experiments, which were generated offline using Llama-3.2-11B-Vision-Instruct and Qwen2.5-VL-7B-Instruct. You can download them from the link below and place them in the ./data/ directory.

# [Link to download pre-generated rationales will be provided here]

🚀 Training and Evaluation
Training and evaluation are handled by a single script, controlled by a configuration file.

Training
To train a new ViLID model, modify a configuration file in the ./configs/ directory to specify your dataset paths and hyperparameters, then run:

python main.py --config_path ./configs/train_mmfakebench_llama.json --mode train

This will start the training process, save checkpoints to the directory specified in your config, and log results.

Evaluation
To evaluate a trained model, use the eval mode and provide the path to the model checkpoint.

python main.py --config_path ./configs/eval_mmfakebench_llama.json --mode eval --eval_model /path/to/your/checkpoint.pth

📈 Results
ViLID establishes a new state-of-the-art on both Fakeddit and MMFakeBench, surpassing strong baselines like MVAE and SNIFFER. Our dual-signal methodology enhances accuracy by 8.4% on Fakeddit and 12.4% on MMFakeBench compared to MVAE.

Ablation studies confirm that all components are crucial. The removal of the alignment loss or the rationale pathway leads to a significant drop in performance, demonstrating their essential role in the model's success.

✍️ Citation
If you find this work useful for your research, please cite our paper:

@inproceedings{ViLID2026,
  title={{ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation}},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
