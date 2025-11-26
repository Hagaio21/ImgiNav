# Project Plan: Guided Floorplan Generation for Navigation in 3D Scenes

**Student:** Hagai Ofer (s233249)
**Advisors:**
* Theodora Kontogianni, Assistant Professor at DTU Compute
* J. Andreas Bærentzen, Professor at DTU Compute

**Institution:** Technical University of Denmark (DTU Compute)
**Program:** MSc in Autonomous Systems
**ECTS:** 30
**Period:** 01 September 2025 to 01 February 2026

---

## 1. Project Description

### 1.1 Background
Autonomous agents navigating unknown indoor environments rely on incomplete, noisy observations (e.g., RGB-D frames, LiDAR scans, or sparse point clouds). Humans handle this by forming and refining mental maps: we infer likely room adjacencies (e.g., kitchen next to living room) and update those hypotheses when new evidence arrives.

This project investigates whether a generative model can play a similar role: produce an initial floorplan hypothesis from sparse, multimodal cues and natural language descriptions, then iteratively refine that hypothesis as the agent explores. The envisioned pipeline is symbolic-to-semantic: high-level, symbolic inputs (room names, adjacency hints) condition a generative diffusion model that proposes a plausible, metrically grounded 2D floorplan. Classical planning (e.g., $A^{*}$) then uses the current floorplan hypothesis for navigation, while observations from partial point clouds feed back to revise the plan. This closes a loop between imagination (generation) and action (navigation), converging toward the real layout over time.

### 1.2 Prior Work
Relevant strands of prior work include:
1.  Scene and layout generation with various generative models.
2.  Symbolic grounding and text-layout alignment.
3.  Classical planning on partial maps.
4.  Data resources for indoor environments (e.g., 3D-FRONT, ScanNet).

The field of generative scene synthesis has evolved from deep convolutional models to autoregressive transformers and denoising diffusion models, which have shown remarkable quality in synthesizing complex indoor layouts. While methods like Holodeck use language to guide the generation of 3D environments for downstream robotics tasks, they typically focus on single-shot generation of complete, static scenes. This project distinguishes itself by focusing on the iterative refinement loop, addressing navigation under partial observability.

### 1.3 Research Question / Hypothesis / Problem Statement

**Problem Statement:**
Given sparse, noisy partial observations and a brief symbolic description of an indoor environment, generate and continually refine a metrically consistent 2D floorplan suitable for downstream navigation.

**Research Questions:**
1.  Can a diffusion-based, symbolically conditioned generator produce navigable floorplan hypotheses from partial observations?
2.  Does iterative evidence assimilation (from new observations) reliably improve plan quality for navigation (path validity, success rate, path stretch)?

**Hypothesis:**
A symbolically conditioned diffusion model, coupled with an observation-update loop, will (a) produce viable initial floorplan hypotheses from sparse cues and (b) monotonically improve navigability metrics as more observations are integrated.

### 1.4 Research Goals and Methods

**Core Research Goals:**
* **G1: Data & Representations.** Prepare training pairs linking symbolic prompts and partial observations to 2D floorplans. Use lightweight 2D projections derived from 3D assets (e.g., room masks and semantic labels), normalized to $[-1, 1]$.
* **G2: Generative Backbone.** Adapt a modular diffusion pipeline with encoder, conditional denoiser (U-Net), noise scheduler, and sampler. Support symbolic/text tokens and observation channels as conditioning.
* **G3: Iterative Refinement.** Design an update rule that incorporates newly observed geometry (e.g., doorways) and re-conditions generation to revise the floorplan hypothesis while preserving global consistency.
* **G5: Evaluation.** Report map quality using metrics benchmarked against relevant literature. Evaluation will emphasize the generated map's utility for navigation (e.g., path validity, success rate) and refinement gains ($\Delta$ metrics per update).

**Stretch Goals:**
* **G4: Navigation Loop.** Integrate a classical planner (e.g., $A^{*}$) to directly evaluate the floorplan's utility in a downstream task by measuring navigation success under the initial hypothesis versus after $k$ refinement rounds.

### 1.5 Empirical Considerations
* **Datasets:** Use existing indoor scene datasets with room semantics (e.g., 3D-FRONT, ScanNet-derived layouts).
* **Compute:** Train at modest resolutions (e.g., 64-256 px) with batch sizes matched to a single GPU.
* **Software:** Python toolchain, modular training/evaluation scripts, and experiment logging.
* **Baselines:** (i) Single-shot generator without iterative updates; (ii) ablation of symbolic conditioning; (iii) comparison against a simpler update rule.

---

## 2. Learning Objectives

### 2.1 Overall Aim
The aim is to design and evaluate a system for guided floorplan generation that supports autonomous robot navigation in unknown indoor environments, generating plausible layouts from sparse observations and refining them as exploration progresses.

### 2.2 Specific Learning Objectives
Through this project, the student will develop the following skills:
* **Generative modelling:** Apply and adapt state-of-the-art diffusion models for structured 2D/3D scene generation.
* **Data processing:** Reconstruct and preprocess 3D indoor scene datasets (3D-FRONT, ScanNet).
* **Semantic reasoning:** Encode symbolic assumptions (e.g., room types, adjacency rules) into generative models.
* **Experimental skills:** Evaluate models quantitatively and qualitatively using appropriate metrics and ablation studies.
* **Research communication:** Document results in a scientific thesis report and public code repository.
* **Robotics integration (Optional):** Implement planning algorithms (e.g., $A^{*}$) on the generated layouts.

### 2.3 Relation to the Study Program
This project contributes to the MSc in Autonomous Systems by combining machine learning, computer vision, and robotics. It integrates generative AI techniques with robotic planning, advancing theoretical understanding and practical skills in autonomous system design.

---

## 3. Project Plan

### 3.1 Milestones

| # | Title | Description | Type | Date |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | Plan | Project plan approved | A (Approval) | 2025-10-01 |
| **M2** | Access | HPC + datasets ready | D (Deadline) | 2025-10-10 |
| **M3** | Data | Preprocessing pipeline done | D (Deadline) | 2025-10-31 |
| **M4** | Gen. MVP | First diffusion results | D (Deadline) | 2025-11-15 |
| **M5** | Refine | Update loop prototype | D (Deadline) | 2025-12-05 |
| **M6** | Planner (Optional) | E2E demo if time permits | D (Deadline) | 2025-12-15 |
| **M7** | Results | Experiments complete | D (Deadline) | 2026-01-10 |
| **F** | Thesis | Final submission | TH (Thesis) | 2026-02-01 |

### 3.2 Deliverables

| # | Title | Type | Date |
| :--- | :--- | :--- | :--- |
| **D1** | Project Plan (this document) | Report | 2025-10-01 |
| **D2** | Master's Thesis (final submission) | Report | 2026-02-01 |
| **D3** | Code repository, models checkpoints | Code | 2026-02-01 |
| **D4** | Public code repository (GitHub) | Code | 2026-02-01 |

### 3.3 Risk Analysis

| # | Title | Risk | Mitigation | Level (1-5) |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | HPC | Access delays | Use local GPU; request temp quota | 3 |
| **R2** | Dataset | Permissions slow | Start with public subsets | 2 |
| **R3** | Data | Noisy/incomplete | Cleaning, filtering, spot checks | 3 |
| **R4** | Train | Instability | Lower res, tune scheduler | 3 |
| **R5** | Cond. | Weak effect | Scale/adapter ablations | 3 |
| **R6** | Planner | Map mismatch | Postprocess cleanup | 2 |
| **R7** | Scope | Too many tasks | Freeze scope at MVP, make navigation optional | 3 |
| **R8** | Time | Overrun | Cut optional experiments | 3 |

### 3.4 Gantt Chart Summary
* **WP0: Prep (lit., HPC, data):** Sept
* **WP1: Dataset pipeline:** Sept - Oct
* **WP2: Generative backbone:** Oct - Nov
* **WP3: Refinement + planner:** Nov - Dec
* **WP4: Experiments:** Dec - Jan
* **WP5: Writing + wrap-up:** Dec - Feb

---

## Bibliography
1.  Peter E. Hart et al. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths". IEEE TSSC (1968).
2.  Huan-ang Fu et al. "3D-FRONT: A Large-Scale 3D Furnished Rooms Dataset". ICCV (2021).
3.  Angela Dai et al. "ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes". CVPR (2017).
4.  Daniel Ritchie et al. "Fast and Flexible Indoor Scene Synthesis via Deep Convolutional Generative Models". arXiv (2018).
5.  Iro Laina Paschalidis et al. "ATISS: Autoregressive Transformers for Indoor Scene Synthesis". NeurIPS (2021).
6.  Jia-Peng Tang et al. "DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis". CVPR (2023).
7.  Xiang Meng et al. "SceneGen: Single Image to 3D Scene Generation". arXiv (2025).
8.  Guanzhi Huang et al. "MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation". CVPR (2025).
9.  Enric Corona et al. "Coherent 3D Scene Diffusion From a Single RGB Image". NeurIPS (2024).
10. Zai-Xin Zhang et al. "SceneHGN: Hierarchical Graph Networks for 3D Indoor Scene Generation...". ICCV (2021).
11. Gege Gao et al. "GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs". CVPR (2024).
12. Sherwin Bahmani et al. "CC3D: Layout-Conditioned Generation of Compositional 3D Scenes". ICCV (2023).
13. Alexey Bokhovkin et al. "SceneFactor: Factored Latent 3D Diffusion...". CVPR (2024).
14. Jonas Schult et al. "ControlRoom3D: Room Generation using Semantic Proxy Rooms". CVPR (2024).
15. Jiaming Sun et al. "LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models". CVPR (2025).
16. Ayush Kadian et al. "LUMINOUS: Indoor Scene Generation for Embodied AI Challenges". arXiv (2021).
17. Andrew Karch et al. "Holodeck: Language Guided Generation of 3D Embodied AI Environments". EMNLP (2022).