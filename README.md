<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViLID GitHub Repository</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #24292e;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
        }
        h1, h2, h3, h4 {
            font-weight: 600;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 24px;
            margin-bottom: 16px;
        }
        h1 {
            font-size: 2em;
        }
        h2 {
            font-size: 1.5em;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
        }
        pre {
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
            word-wrap: normal;
        }
        pre code {
            display: inline;
            padding: 0;
            margin: 0;
            overflow: visible;
            line-height: inherit;
            word-wrap: normal;
            background-color: transparent;
            border: 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }
        .badges img {
            margin-right: 5px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation</h1>

        <div class="badges">
            <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
            <a href="https://www.python.org/downloads/release/python-390/" target="_blank"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
            <a href="https://arxiv.org/abs/2XXX.XXXXX" target="_blank"><img src="https://img.shields.io/badge/arXiv-2XXX.XXXXX-b31b1b.svg" alt="arXiv Paper"></a> <!-- Replace with your arXiv ID -->
        </div>

        <p>This repository contains the official implementation for the paper: <strong>"ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation"</strong>.</p>
        <p>ViLID is a novel framework designed to identify fine-grained semantic misalignments between text and images in multimodal content. It addresses the challenge of detecting nuanced misinformation by integrating direct feature-level inconsistency with higher-order reasoning based on AI-generated rationales.</p>

        <hr>

        <h2>📜 Abstract</h2>
        <p>The proliferation of multimodal misinformation, characterized by the integration of text and images to create deceptive narratives, presents a significant societal challenge. To fill this gap, we introduce ViLID, a novel framework that excels at identifying fine-grained semantic misalignments between text and images. Key contributions include: a principled cross-modal inconsistency score; a rationale-augmented reasoning module utilizing LLMs to produce explicit textual and visual explanations; and a novel alignment regularization term to enhance model robustness. Extensive evaluations on the <strong>Fakeddit</strong> and <strong>MMFakeBench</strong> benchmarks demonstrate that ViLID achieves state-of-the-art performance in misinformation detection.</p>

        <hr>

        <h2>🏗️ Model Architecture</h2>
        <p>ViLID's architecture uses a dual-pathway analysis to detect inconsistencies. As shown in the diagram below, every text-image pair is processed through both a <strong>Non-Reasoning Pathway</strong> (for direct feature extraction) and a <strong>Reasoning Pathway</strong> (which uses LLMs to generate and encode rationales). The features and inconsistency scores from both pathways are then combined in a fusion transformer and passed to a classifier to make the final prediction.</p>
        <img src="https://i.imgur.com/GQqxS3P.png" alt="ViLID Architecture Diagram">
        <p style="text-align: center; font-style: italic;">Figure 1: High-level overview of the ViLID architecture.</p>

        <hr>

        <h2>✨ Key Features</h2>
        <ul>
            <li><strong>Rationale-Augmented Reasoning</strong>: Integrates LLM-generated rationales to capture higher-order semantic inconsistencies between text and image explanations.</li>
            <li><strong>Dual Inconsistency Scores</strong>: Computes two separate scores: <code>S_inc</code> for direct feature mismatch and <code>S_r</code> for rationale-level mismatch, providing the model with explicit alignment signals.</li>
            <li><strong>Alignment Regularization</strong>: A novel loss term (<code>L_align</code>) encourages the model to learn consistent representations for truthful data, improving robustness and generalization.</li>
            <li><strong>State-of-the-Art Performance</strong>: Achieves SOTA performance on challenging benchmarks like Fakeddit and MMFakeBench.</li>
        </ul>

        <hr>

        <h2>⚙️ Setup and Installation</h2>
        <p>To get started, clone the repository and set up the Conda environment using the provided files.</p>
        <pre><code># 1. Clone the repository
git clone https://github.com/your-username/ViLID.git
cd ViLID

# 2. Create and activate the Conda environment
conda create -n vilid python=3.9
conda activate vilid

# 3. Install dependencies
pip install -r requirements.txt</code></pre>

        <hr>

        <h2>📊 Datasets</h2>
        <p>Our experiments use the following datasets. Please follow the instructions from the original sources to download them.</p>
        <ul>
            <li><strong>Fakeddit</strong>: A large-scale multimodal dataset from Reddit. You can find more information in the original paper by <em>Nakamura et al. (2019)</em>. Due to its size, we use a 25% stratified sample for our experiments.</li>
            <li><strong>MMFakeBench</strong>: A modern benchmark designed with mixed-source misinformation. More details are available in the paper by <em>Liu et al. (2024)</em>.</li>
            <li><strong>M3A</strong>: Used for our zero-shot cross-domain evaluation. See <em>Xu et al. (2024)</em> for access details.</li>
        </ul>
        <h4>Pre-generated Rationales</h4>
        <p>Our model relies on pre-generated rationales for training. We provide the rationales used in our experiments, which were generated offline using <code>Llama-3.2-11B-Vision-Instruct</code> and <code>Qwen2.5-VL-7B-Instruct</code>. You can download them from the link below and place them in the <code>./data/</code> directory.</p>
        <pre><code># [Link to download pre-generated rationales will be provided here]</code></pre>

        <hr>

        <h2>🚀 Training and Evaluation</h2>
        <p>Training and evaluation are handled by a single script, controlled by a configuration file.</p>
        <h3>Training</h3>
        <p>To train a new ViLID model, modify a configuration file in the <code>./configs/</code> directory to specify your dataset paths and hyperparameters, then run:</p>
        <pre><code>python main.py --config_path ./configs/train_mmfakebench_llama.json --mode train</code></pre>
        <p>This will start the training process, save checkpoints to the directory specified in your config, and log results.</p>
        <h3>Evaluation</h3>
        <p>To evaluate a trained model, use the <code>eval</code> mode and provide the path to the model checkpoint.</p>
        <pre><code>python main.py --config_path ./configs/eval_mmfakebench_llama.json --mode eval --eval_model /path/to/your/checkpoint.pth</code></pre>

        <hr>

        <h2>📈 Results</h2>
        <p>ViLID establishes a new state-of-the-art on both Fakeddit and MMFakeBench, surpassing strong baselines like MVAE and SNIFFER. Our dual-signal methodology enhances accuracy by <strong>8.4% on Fakeddit</strong> and <strong>12.4% on MMFakeBench</strong> compared to MVAE.</p>
        <p>Ablation studies confirm that all components are crucial. The removal of the <strong>alignment loss</strong> or the <strong>rationale pathway</strong> leads to a significant drop in performance, demonstrating their essential role in the model's success.</p>

        <hr>

        <h2>✍️ Citation</h2>
        <p>If you find this work useful for your research, please cite our paper:</p>
        <pre><code>@inproceedings{ViLID2026,
  title={{ViLID: A Rationale-Enhanced Vision-Language Inconsistency Detector for Multimodal Misinformation}},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}</code></pre>

        <hr>

        <h2>📜 License</h2>
        <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>
    </div>
</body>
</html>
