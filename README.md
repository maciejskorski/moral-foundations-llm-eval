# Estimating Human vs AI Moral Competences

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/maciejskorski)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=WeightsAndBiases&logoColor=white)](https://wandb.ai)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-brightgreen.svg)](https://maciejskorski.github.io/moral-foundations-llm-eval/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.13804-b31b1b.svg)](https://arxiv.org/abs/2508.13804)

## 📋 Overview

Large-scale comprehensive evaluation of LLMs on moral foundation classification using Haidt's Moral Foundations Theory and Bayesian statistical modeling accelerated with GPU. This project systematically compares AI performance against human annotators across moral dimensions.

**🎯 Key findings:** AI models show more balanced predictions and much fewer false negatives (missed findings) compared to human annotators, achieving 75th-100th percentile performance across moral foundations.

AI vs Human Performance

![AI vs Human Performance](docs/results/ai-rank.svg)

AI vs Human Errors

![AI vs Human Errors](docs/results/ai-humans.svg)

## 🗂️ Project Structure

### [`datasets.ipynb`](src/datasets.ipynb)
**📊 Dataset Standardization** 

Standardizes MFRC (Reddit), MFTC (Twitter), and eMFD datasets into unified 5-foundation taxonomy. Handles deduplication, multi-annotator formats, and convert clean datasets to the 🤗 HuggingFace format.

### [`ask_llm.ipynb`](src/ask_llm.ipynb)
**🤖 LLM Evaluation Pipeline**

Evaluates multiple LLMs (Claude-4, DeepSeek-V3, Llama4-Maverick, etc.) on moral foundation classification with:
- Async processing for efficient batch inference
- Standardized prompting across models
- Performance logging to Weights & Biases
- Error handling and result validation

**Usage with Papermill:**
```bash
papermill ask_llm.ipynb output.ipynb -p model_name "claude-4-sonnet" -p test_data 'morality-MFRC' -p sample 100 -p temperature 0.3 --log-output
```

### [`annots_competences.ipynb`](src/annots_competences.ipynb)
**📈 Human vs AI Performance Analysis**

Implements a novel GPU-efficient Dawid-Skene competence model in TensorFlow to:
- Estimate annotator competence and consensus
- Compare AI performance against human baselines
- Generate percentile rankings and balanced accuracy metrics
- Visualize performance distributions across moral dimensions and datasets



## ⚙️ Dependencies

- `datasets`, `transformers` for data handling
- `anthropic`, `openai`, `replicate` for LLM APIs
- `tensorflow`, `tensorflow-probability` for competence modeling
- `wandb` for experiment tracking
- `papermill` for parameterized execution
