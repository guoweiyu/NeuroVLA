# 🚀 LIBERO Evaluation

This document explains how to evaluate **NeuroVLA** on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark. The pipeline has two parts:

1. Setting up the `LIBERO` environment and dependencies.
2. Running the evaluation as a **client/server pair**: a NeuroVLA policy server plus the LIBERO simulation client.

We have verified that this workflow runs on **NVIDIA A100 / A800** and **RTX 4090** GPUs.

---

## 📊 Experimental Results

NeuroVLA on LIBERO (success rate %, 4 task suites, 100 episodes per suite). The spiking action head is evaluated with three neuron models of increasing biological richness:

| SNN action head | Spatial | Object | Goal | Long (10) | Avg  |
|:----------------|:-------:|:------:|:----:|:---------:|:----:|
| LIF             |   94    |   99   |  98  |    85     | 94.0 |
| ALIF            | **96**  |   98   |  96  |    88     | 94.5 |
| **PLIF** (best) |   90    |   99   |  96  |  **95**   | **95.0** |

**Learning rule (batch 32, 50k steps).** A biologically-plausible, local **e-prop** rule (eligibility traces, O(1)-in-time memory, no backprop-through-time) matches surrogate-gradient BPTT:

| Learning rule | Spatial | Object | Goal | Long (10) | Avg   |
|:--------------|:-------:|:------:|:----:|:---------:|:-----:|
| BPTT          |   88    |   98   |  98  |    91     | 93.75 |
| **e-prop**    |   93    |   99   | 100  |    90     | **95.5** |

> Full training/eval recipes and the broader leaderboard are maintained in [AlphaBrain ▸ NeuroVLA quickstart](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/docs/quickstart/neurovla.md).

---

## ⬇️ 0. Checkpoints

Train your own checkpoint with the [training pipeline](#-libero-training) below, or use the maintained models from the AlphaBrain model hub:

- 🤗 [huggingface.co/AlphaBrainGroup](https://huggingface.co/AlphaBrainGroup)

Place (or point to) your checkpoint, then set its path inside `examples/LIBERO/run_server.sh`.

---

## 📦 1. Environment Setup

First follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` simulation environment. The NeuroVLA policy server runs in the `neurovla` environment (see the [top-level README](../../README.md#%EF%B8%8F-installation) for setup).

---

## 🚀 2. Evaluation Workflow

Run **from the repository root** using **two terminals**, one per environment:

- **`neurovla` environment** — runs the NeuroVLA inference server.
- **`LIBERO` environment** — runs the simulation client.

### Step 1. Start the policy server (`neurovla` environment)

```bash
bash examples/LIBERO/run_server.sh
```

⚠️ Set the correct checkpoint path inside `examples/LIBERO/run_server.sh`.

---

### Step 2. Start the simulation (`LIBERO` environment)

```bash
bash examples/LIBERO/eval_libero.sh
```

⚠️ Set the matching checkpoint path inside `examples/LIBERO/eval_libero.sh` so the correct action-unnormalization stats are loaded.

---

# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset

Download the datasets into `playground/Datasets/LEROBOT_LIBERO_DATA/`:

- [LIBERO-Spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-Object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-Goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

## 🚀 Step 1: Start training

```bash
bash scripts/run_scripts/run_libero_train.sh
```

> Set the dataset/model paths inside the script before running. For the full parameterized pipeline (pretrain → R-STDP fine-tune → eval with online STDP), see the [AlphaBrain brain-inspired scripts](https://github.com/AlphaBrainGroup/AlphaBrain/tree/main/scripts/run_brain_inspired_scripts).
