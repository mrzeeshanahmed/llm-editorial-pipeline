# 📰 High-Throughput Editorial Analyzer (LLM Pipeline)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Gemma--3--1B-orange)](https://huggingface.co/Zap11/gemma-3-1b-editorial-analyzer)
[![Infrastructure](https://img.shields.io/badge/Infrastructure-CPU%20Inference-green)]()
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20LlamaCPP-purple)]()

An end-to-end MLOps pipeline designed to fine-tune, optimize, and deploy a lightweight LLM (**Gemma-3-1B**) for real-time editorial analysis on commodity CPU hardware.

This project addresses the challenge of processing **20+ news articles every 5 minutes** within a strict compute budget (2 vCPU / 16GB RAM) by leveraging **LoRA fine-tuning**, **synthetic data augmentation**, and a novel **"Headless" schema-agnostic inference strategy**.

---

##  🏗️ Architecture Overview

The pipeline decouples reasoning (LLM) from formatting (Application Layer) to maximize throughput.



    A[Raw Articles] -->|Cleaning & Balancing| B(Synthetic Data Engine)
    B -->|Augmented Dataset| C{Unsloth Trainer}
    C -->|LoRA Fine-Tuning| D[Gemma-3-1B Adapter]
    D -->|Merge & Quantize| E[GGUF Q4_K_M]
    E -->|Deploy| F[FastAPI / CPU Inference]
    F -->|Headless Stream| G[Python Schema Reconstructor]
    G -->|Final JSON| H[Supabase DB]


---
## 🚀 Key Engineering Features
1. Custom LoRA Fine-Tuning (Memory Optimized)
Constraint: Training on a consumer GPU (RTX 3050 Ti, 4GB VRAM).
Solution: Utilized Unsloth for memory-efficient backpropagation and LoRA (Low-Rank Adaptation) with r=8, alpha=16.
Outcome: Successfully fine-tuned a 1.1B parameter model with a batch size of 1 (simulated batch size 8 via gradient accumulation) without OOM errors.

2. Synthetic Data Augmentation (SDV Principles)
Problem: The original dataset (525 rows) was heavily biased towards "Neutral" political articles, leading to poor recall on "Biased" content.
Solution: Implemented a programmatic data synthesis strategy inspired by Synthetic Data Vault (SDV).
Oversampling: Statistically boosted under-represented classes (e.g., Category=Education, Bias=True) by 4x.
Result: Balanced dataset of 1,021 rows, significantly improving the model's F1-score on rare edge cases.

3. "Headless" Schema-Agnostic Inference
Innovation: Traditional LLMs waste ~40% of tokens generating JSON syntax ({ "key": "value" }).
Strategy: The model was trained to output a compressed ordered text stream separated by delimiters (|||).
Standard: {"sentiment": 5, "bias": true} (12 tokens)
Headless: 5 ||| True (3 tokens)
Benefit: Reduced inference latency by ~35% on CPU and decoupled the database schema from the model weights.
---
## 📂 Repository Structure



    llm-editorial-pipeline/
    ├── README.md                 # Project Documentation
    ├── training/
    │   ├── build_dataset.py      # Data balancing & augmentation logic
    │   ├── train_local.py        # Unsloth/LoRA fine-tuning script
    │   └── merge_checkpoints.py  # Script to merge LoRA adapters ("Model Soup")
    ├── inference/
    │   ├── app.py                # FastAPI service with schema reconstruction
    │   └── Dockerfile            # CPU-optimized container (llama-cpp-python)
    └── requirements.txt          # Python dependencies

---

## 🛠️ Reproduction Steps


  ##### Phase 1: Data Engineering
Generate the balanced training dataset from raw CSV exports. (Find in HF)

```bash
`python training/balance_dataset.py`
```
 Output: training/training_dataset_balanced.jsonl (1000+ rows)
 
##### Phase 2: Fine-Tuning (Local GPU)
Run the quantized training pipeline.

```bash
python training/train_local.py
```
##### Base Model: unsloth/gemma-3-1b-it-bnb-4bit

Epochs: 3.5 (Early stopping based on loss convergence ~0.8)

Output: Generates model.gguf (Q4_K_M quantization).

##### Phase 3: Deployment (CPU)
Build and run the inference API container.

```bash
cd inference
docker build -t news-analyzer .
docker run -p 7860:7860 news-analyzer
```
Test the API:

```bash
curl -X POST "http://localhost:7860/analyze" \
     -H "Content-Type: application/json" \
     -d '{"id": "123", "content": "The government announced a new tax policy..."}'
```
---

##### 📊 Performance Metrics
| Metric                  | Zero-Shot Baseline | Fine-Tuned (Headless)   |
| ----------------------- | ------------------ | ----------------------- |
| Model Size              | 2.5 GB (FP16)      | **768 MB (Q4_K_M)**     |
| Inference Speed (CPU)   | ~12 tok/sec        | **~18 tok/sec**         |
| Bias Detection Accuracy | 62%                | **89%**                 |
| Throughput              | 8 articles / 5 min | **22 articles / 5 min** |

---
### 🧠 Model Weights
The final quantized model is hosted on Hugging Face and is licensed under Apache 2.0.

👉 [Download Gemma-3-1B-Editorial-Analyzer](https://huggingface.co/Zap11/gemma-3-1b-editorial-analyzer "Download Gemma-3-1B-Editorial-Analyzer")

---

### 📜 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.


