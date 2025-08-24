# (Pose-aware IR)

##  Overview
This project implements an **end-to-end pipeline for colonoscopy place recognition** using:
- **Model**: ResNet-101 + GeM pooling
- **Loss**: Pose-aware InfoNCE (contrastive learning)
- **Datasets**:
  - Pretrain: **SimCol3D** (synthetic data with ground-truth pose)
  - Fine-tune: **C3VD** (phantom colonoscope dataset)
  - Test: **Colon10K** (real patient benchmark with annotated matches)

The pipeline trains on SimCol3D, fine-tunes on C3VD, and evaluates on Colon10K.

---

