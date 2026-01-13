<div align="center">

# TopoFreeRL: Topology-Free Reinforcement Learning for Distributed LLM Inference

<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

---

## Abstract

**TopoFreeRL** is a lightweight, high-performance reinforcement learning framework designed for real-time **LLM serving** and **MoE inference** in edge-cloud networks.

While Graph Neural Networks (GNNs) are powerful, they are often too computationally expensive for real-time decision-making on the edge. TopoFreeRL solves this by introducing a **"Graph-Free" perception module** that efficiently captures critical network bottlenecks without the heavy overhead of GNNs.

### Key Highlights
* **Ultra-Fast Decision Making**: Replaces complex message-passing with a linear-complexity **Squeeze-and-Excitation attention mechanism**, enabling scalable, real-time scheduling even under bursty traffic.
* **Dynamic Adaptation**: Features a **Dynamic Weight Adaptation (DWA)** mechanism that autonomously adjusts optimization priorities to navigate unstable, non-stationary network environments.
* **State-of-the-Art Performance**: Significantly outperforms existing baselines, reducing network transmission costs by **97.0%** and boosting inference efficiency by **24.4%**, all while maintaining robust performance in unseen environments.

---

## News

- Code and sample dataset (Server1_Trap) released.

---
## TopoFreeRL at a Glance

Although Graph Neural Networks (GNNs) can capture structural information for routing/scheduling in large-scale LLM serving, their message-passing inference introduces non-trivial runtime overhead, which becomes prohibitive for real-time edge decision-making under bursty traffic. **TopoFreeRL** is a **graph-free** reinforcement learning approach that preserves **topology awareness** while maintaining **low-latency** control.

### Key Ideas

**(1) Graph-Free Perception with Preference-Gated SE Attention.**  
Instead of message passing on dynamic graphs, TopoFreeRL directly consumes raw telemetry and applies a lightweight **preference-gated Squeeze-and-Excitation (SE)** attention module to amplify bottleneck-related channels with **linear complexity**, effectively avoiding GNN scalability limits.

**(2) Dynamic Weight Adaptation (DWA) for Non-Stationary Pareto Navigation.**  
To track the Pareto frontier under time-varying workloads, TopoFreeRL introduces **Dynamic Weight Adaptation (DWA)**, which autonomously recalibrates multi-objective preferences using **entropy-regularized metric drift**, improving stability and robustness without manual retuning.

**(3) Strong Efficiency and Generalization.**  
Across large-scale heterogeneous topologies derived from real-world traces, TopoFreeRL delivers substantial system-level gains: **97.0% lower network transmission costs** and **24.4% higher comprehensive inference efficiency**, while preserving **zero-shot generalization** with **<10% degradation** in unseen hyperscale environments.

---

## Training Curves (Examples)

> Replace the image paths below with your repository paths (e.g., `assets/figs/...`).

<p align="center">
  <img src="results/Fig1(a) Comparison_Reward.png" width="32%" alt="Training return / objective improvement" />
  <img src="results/Fig1(b) Comparison_Latency.png" width="32%" alt="Policy entropy / exploration dynamics" />
  <img src="results/Fig1(c) Comparison_Cost.png" width="32%" alt="Dynamic Weight Adaptation (DWA) weights over time" />
</p>

**Figure A — Objective improvement / return.** Demonstrates convergence behavior and overall optimization progress during training.  
**Figure B — Exploration dynamics (entropy or KL).** Highlights stable exploration and learning dynamics under bursty traffic.  
**Figure C — DWA preference weights.** Visualizes how DWA automatically shifts optimization priorities (e.g., latency vs. cost vs. risk) in response to non-stationary metric drift.


## Project Structure

```
.
├── TopoFreeRL/             # [Ours] TopoFreeRL Algorithm Implementation
│   ├── agent.py            # PPO Agent with Spatiotemporal Awareness
│   ├── model.py            # Actor-Critic Networks
│   └── train.py            # Training Loop
├── PFAPPO/                 # [Baseline] PF-PPO Implementation
├── PPO_algorithm/          # [Baseline] Standard PPO Implementation
├── Stark_Scheduler/        # [Baseline] STARK Implementation
├── env.py                  # Simulation Environment (Edge-Cloud MoE)
├── data1/                  # Dataset Directory
│   └── Server1_Trap/       # Provided Sample Dataset (500 Nodes)
├── total/                  # Visualization & Analysis Scripts
└── requirements.txt        # Dependencies
```

---

## Dataset: Edge-Cloud MoE Bench

We introduce a high-fidelity benchmark dataset for edge-cloud MoE inference scheduling. 

### Why this dataset is valuable?
Unlike synthetic traces used in prior works, this dataset captures the complexity of real-world edge computing:
1.  **Heterogeneity**: Diverse server compute capabilities and cost models.
2.  **Geo-Distribution**: Latency based on physical distance (Haversine formula).
3.  **Adversarial "Traps"**: Specific network links exhibit stochastic high latency/packet loss, simulating real-world network jitter. This is critical for testing robustness.
4.  **MoE Workflows**: Tasks require sequential processing by specific expert models distributed across the topology.

### Availability
We provide the **`Server1_Trap` (500 servers)** scale as a sample for reproducibility.

> **Note**: The full dataset includes larger scales (1000 and 2000 servers) with varying densities and trap configurations. Due to the size and commercial value of the full topology data, only the sample is open-sourced. 
>
> Researchers requiring the **complete dataset** for comparison benchmarks should contact: **gymorsiback@tju.edu.cn**

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anonymous/TopoFreeRL.git
   cd TopoFreeRL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### 1. Training

To train TopoFreeRL on the provided sample dataset:

```bash
python TopoFreeRL/train.py \
    --data data1 \
    --regions Server1_Trap \
    --epochs 100 \
    --device cuda
```

You can also train baseline algorithms (e.g., PF-PPO, Standard PPO):

```bash
# Train PF-PPO
python PFAPPO/train.py --data data1 --regions Server1_Trap --epochs 100
```

### 2. Inference & Evaluation

Load the trained model to evaluate performance:

```bash
python TopoFreeRL/inference.py \
    --data data1 \
    --region Server1_Trap \
    --model results/TopoFreeRL/models/LATEST_Server1_Trap_seed42_final.pt \
    --episodes 500
```

### 3. Visualization

Generate comparison figures (Learning Curves, Cost Breakdown, Latency CDFs):

```bash
# Generates all figures in the 'total/' folder
python total/plot_all_comparison.py
```

---

## Results

Our method consistently outperforms baselines in terms of reward, latency, and cost efficiency, especially in "Trap" environments.

*(Visualization results are generated in the `total/` directory)*

---

## Acknowledgements & References

This repository is **inspired by** (and in some components may adapt design ideas from) the following representative works.  
Unless explicitly stated, this project is an **independent implementation** and is **not affiliated with** the original authors.

### Scheduling / Multi-cloud / SAGIN
- **A3C-based workflow scheduling (multi-cloud)**: Tang *et al.* “Workflow scheduling based on asynchronous advantage actor–critic algorithm in multi-cloud environment.” *Expert Systems with Applications*, 2024. DOI: 10.1016/j.eswa.2024.125245. :contentReference[oaicite:0]{index=0}  
- **PFAPPO (fairness-aware scheduling in SAGIN)**: Sun *et al.* “Proportional Fairness-Aware Task Scheduling in Space-Air-Ground Integrated Networks.” *IEEE Transactions on Services Computing*, 17(6):4125–4137, 2024. DOI: 10.1109/TSC.2024.3478730. :contentReference[oaicite:1]{index=1}  
- **Graph-assisted offline–online DRL for dynamic workflow scheduling (GOODRL)**: Yang *et al.* “Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling.” *ICLR 2025*. :contentReference[oaicite:2]{index=2}  

### Transformer / Tracking / Routing
- **Equity-Transformer (min–max routing)**: Son *et al.* “Equity-Transformer: Solving NP-hard Min-Max Routing Problems as Sequential Generation with Equity Context.” arXiv:2306.02689 (AAAI 2024 version noted on arXiv). DOI: 10.48550/arXiv.2306.02689. :contentReference[oaicite:3]{index=3}  
- **STARK (spatio-temporal transformer tracking)**: Yan *et al.* “Learning Spatio-Temporal Transformer for Visual Tracking.” *ICCV 2021*, pp. 10448–10457. :contentReference[oaicite:4]{index=4}  

### PPO variants / Exploration
- **Colored Noise in PPO (correlated action sampling)**: Hollenstein *et al.* “Colored Noise in PPO: Improved Exploration and Performance through Correlated Action Sampling.” arXiv:2312.11091 (also appears as an AAAI paper per bibliographic indexes). :contentReference[oaicite:5]{index=5}  

### Foundational algorithms (if used)
- **A3C (original)**: Mnih *et al.* “Asynchronous Methods for Deep Reinforcement Learning.” arXiv:1602.01783 / ICML 2016. :contentReference[oaicite:6]{index=6}  
- **PPO (original)**: Schulman *et al.* “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017. :contentReference[oaicite:7]{index=7}  

> Note: If this repository reuses any third-party code/models, please also follow the corresponding licenses and provide code-level attribution (e.g., NOTICE file, link to the original repository, and explicit description of reused components).


@article{Tang2024MCWSA3C,
  title   = {Workflow scheduling based on asynchronous advantage actor--critic algorithm in multi-cloud environment},
  author  = {Tang, Xuhao and Liu, Fagui and Wang, Bin and Xu, Dishi and Jiang, Jun and Wu, Qingbo and Chen, C. L. Philip},
  journal = {Expert Systems with Applications},
  year    = {2024},
  doi     = {10.1016/j.eswa.2024.125245}
}

@article{Sun2024PFAPPO,
  title   = {Proportional Fairness-Aware Task Scheduling in Space-Air-Ground Integrated Networks},
  author  = {Sun, Gang and Wang, Yuhui and Yu, Hongfang and Guizani, Mohsen},
  journal = {IEEE Transactions on Services Computing},
  volume  = {17},
  number  = {6},
  pages   = {4125--4137},
  year    = {2024},
  doi     = {10.1109/TSC.2024.3478730}
}

@inproceedings{Yan2021STARK,
  title     = {Learning Spatio-Temporal Transformer for Visual Tracking},
  author    = {Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10448--10457},
  year      = {2021}
}

@misc{Son2024EquityTransformer,
  title         = {Equity-Transformer: Solving NP-hard Min-Max Routing Problems as Sequential Generation with Equity Context},
  author        = {Son, Jiwoo and Kim, Minsu and Choi, Sanghyeok and Kim, Hyeonah and Park, Jinkyoo},
  year          = {2024},
  eprint        = {2306.02689},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2306.02689}
}

@misc{Hollenstein2023ColoredNoisePPO,
  title         = {Colored Noise in PPO: Improved Exploration and Performance through Correlated Action Sampling},
  author        = {Hollenstein, Jakob and Martius, Georg and Piater, Justus},
  year          = {2023},
  eprint        = {2312.11091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@inproceedings{Yang2025GOODRL,
  title     = {Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling},
  author    = {Yang, Yifan and Chen, Gang and Ma, Hui and Zhang, Cong and Cao, Zhiguang and Zhang, Mengjie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@misc{Mnih2016A3C,
  title         = {Asynchronous Methods for Deep Reinforcement Learning},
  author        = {Mnih, Volodymyr and Badia, Adria Puigdomenech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P. and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  year          = {2016},
  eprint        = {1602.01783},
  archivePrefix = {arXiv}
}

@misc{Schulman2017PPO,
  title         = {Proximal Policy Optimization Algorithms},
  author        = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  year          = {2017},
  eprint        = {1707.06347},
  archivePrefix = {arXiv}
}




## Contact

For any questions regarding the code or to request the **full dataset**, please email:  
**gymorsiback@tju.edu.cn**
