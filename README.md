<div align="center">

# NeuroVLA: A Brain-like Embodied Intelligence for Fluid and Fast Reflexive Robotics Control

**Weiyu Guo**<sup>1,4</sup>, **He Zhang**<sup>1,4</sup>, **Pengteng Li**<sup>1,4</sup>, **Tiefu Cai**<sup>1,4</sup>, **Ziyang Chen**<sup>1,4</sup>, **Yandong Guo**<sup>1,4</sup>, <br>
**He Xiao**<sup>4</sup>, **Yongkui Yang**<sup>3</sup>\*, **Ying Sun**<sup>1,2</sup>\*, **Hui Xiong**<sup>1,2</sup>\*

<sup>1</sup>The Thrust of Artificial Intelligence, HKUST (Guangzhou), China  
<sup>2</sup>The Department of CSE, HKUST, Hong Kong, China  
<sup>3</sup>Shenzhen Institutes of Advanced Technology, CAS, China  
<sup>4</sup>AI<sup>2</sup>Robotics, Shenzhen, China  


[![Nature](https://img.shields.io/badge/Nature-Under%20Review-E30613?style=flat-square)](https://github.com/guoweiyu/NeuroVLA)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-AGPL%20v3-00599C?style=flat-square&logo=gnu-bash&logoColor=white)](https://www.gnu.org/licenses/agpl-3.0)

</div>


<br>
<div align="center">
  <table style="border: none; border-collapse: collapse; width: 100%;">
    <tr>
      <td align="center" style="border: none; padding: 5px; width: 33%;">
        <img src="assets/pourwater.gif" width="100%" alt="Fine-grained Manipulation">
        <br>
        <sub><b>(a) Fine-grained Manipulation</b><br>Precision pouring task demanding high-frequency feedback for robotic stable control.</sub>
      </td>
      <td align="center" style="border: none; padding: 5px; width: 33%;">
        <img src="assets/shake.gif" width="100%" alt="Temporal Memory">
        <br>
        <sub><b>(b) Temporal Memory</b><br>Multi-stage shaking task demonstrating phase tracking and working memory capabilities.</sub>
      </td>
      <td align="center" style="border: none; padding: 5px; width: 33%;">
        <img src="assets/collision.gif" width="100%" alt="Reflexive Safety">
        <br>
        <sub><b>(c) Reflexive Safety</b><br>Rapid withdrawal and recovery upon unexpected collision to ensure hardware safety.</sub>
      </td>
    </tr>
  </table>
  <p><em>Real-world experiments evaluating precision (Pouring), memory (Shaking), and safety reflexes (Collision Recovery).</em></p>
</div>
<br>


## 📖 Overview

The pursuit of general-purpose embodied intelligence faces a critical **sensorimotor paradox**: traditional Vision-Language-Action (VLA) models suffer from "temporal blindness" and high latency, leading to action jitter and an inability to reflex instantaneously in dynamic scenarios.

**NeuroVLA** introduces a bio-inspired, tri-level hierarchical architecture that restores the canonical division of labor found in biological motor systems. Instead of a monolithic processor, NeuroVLA decouples high-level cognition from low-level motor control:

1.  **Cortical Module (Vision-Language):** Responsible for semantic planning and high-level goal generation.
2.  **Cerebellar Module (Adaptive):** Functions as a high-frequency adaptive filter to predict sensory consequences and refine timing.
3.  **Spinal Module (Spiking Neural Network):** Implements asynchronous, localized actuation and fast sensorimotor loops.

By mapping the spinal module to event-driven spiking networks, NeuroVLA exploits temporal sparsity to minimize end-to-end latency, enabling localized, hardware-efficient learning on edge devices.

## 🔥 Key Results

Our experiments on both simulated benchmarks and physical robotic hardware demonstrate distinctive capabilities that purely scaling monolithic VLAs cannot replicate:

* **Kinematic Smoothness (75% Jerk Reduction):** The cerebellar module functions as an adaptive filter, effectively suppressing high-frequency intention tremor. This reduces kinematic jerk by over **75%**, ensuring fluid execution even with noisy visual feedback.
* **Survival Reflexes (< 20 ms Latency):** Under unexpected physical collisions, the cerebellar-spinal loops trigger rapid withdrawal reflexes in **< 20 ms**, bypassing the prohibitive latency (> 200 ms) of the cortical loop to protect hardware.
* **Emergent Sparsity:** The neuromorphic spinal layer exhibits unsupervised functional self-organization without explicit training signals:
    * *Temporal Sparsity:* Neurons spontaneously revert to quiescence during static posturing to minimize metabolic cost.
    * *Spatial Disentanglement:* The network naturally segregates high-dimensional control signals into distinct, somatotopic behavioral modes.

## 🛠️ Installation

The environment setup is based on standard VLA dependencies. We recommend using `conda` to manage the environment.

### Prerequisites
* Linux (Ubuntu 20.04/22.04 recommended)
* Python 3.10+
* NVIDIA GPU with CUDA support

### Environment Setup

```bash
# 1. Create a conda environment
conda create -n neurovla python=3.10 -y
conda activate neurovla

# 2. Install PyTorch (Adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation
```

Note: For specific dependency versions and detailed configuration related to the base VLA framework, please refer to the [StarVLA](https://github.com/starVLA/starVLA) Environment Setup Guide. Our implementation builds upon these foundational libraries.

## 🚀 Usage

```bash
# 1. Run training example
bash NeuroVLA/scripts/run_scripts/run_libero_train_NeuroVLA.sh

# 2. Run evaluation example
bash NeuroVLA/examples/LIBERO/eval_libero.sh
```

## 📝 Citation
If you find our code or architecture helpful in your research, please cite our repository:

```bibtex
@misc{guo2025neurovla,
  author = {Guo, Weiyu and Zhang, He and Li, Pengteng and Cai, Tiefu and Chen, Ziyang and Guo, Yandong and Xiao, He and Yang, Yongkui and Sun, Ying and Xiong, Hui},
  title = {NeuroVLA: A Brain-like Embodied Intelligence for Fluid and Fast Reflexive Robotics Control},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/guoweiyu/NeuroVLA}}
}
```

## 🛡️ License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

This is a strict copyleft license. If you use this software (or a modified version of it) to provide a service over a network, you must make the source code available to the users of that service.

See LICENSE for more details.
