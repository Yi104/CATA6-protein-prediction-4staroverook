
# 🧬 CAFA6 Protein Function Prediction Project Plan

**Kaggle Competition:**  
https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview

---

## 1. Project Goal (The Why)

Proteins are the central movers in biology, with their function determined by their amino acid sequence.  
Our goal is to build a model that **predicts Gene Ontology (GO) functions** — including:

- **Molecular Function (MF)**
- **Biological Process (BP)**
- **Cellular Component (CC)**

— directly from raw protein sequences.

**Why this matters:**  
Accurate function prediction accelerates biomedical discovery and enables the development of new treatments.

---

## 2. Core Problem: Multi-Label Protein Function Prediction

**Input:**  
Raw amino acid sequence (FASTA)

**Output:**  
A set of GO terms + probability scores across MF, BP, CC.

**Challenges:**

- Proteins have **multiple functions**
- GO hierarchy (is_a, part_of relationships)
- Labels are noisy and incomplete
- Prospective evaluation: test labels will be **discovered after** the competition deadline

---

## 3. Evaluation Summary (Fmax)

The official competition metric is **Fmax**, defined as:

- Maximum F1-score over all thresholds τ
- Weighted by GO term information content
- Averaged over **MF**, **BP**, and **CC**

This emphasizes correct prediction of **specific, informative GO terms**.

---

# 4. Project Timeline (8 Weeks / 4 Sprints)

| Metric | Target | Notes |
|--------|--------|-------|
| **Primary: Fmax** | **> 0.65** | Main CAFA6 ranking metric |
| **Secondary: MF F1-score** | > 0.75 | MF typically performs best |
| **Deliverable** | `predictions.tsv` | Must follow CAFA6 formatting rules |

---

# 5. Team Roles & Platforms

| Role | Initials | Status | Expertise | Platform | Focus |
|------|----------|--------|-----------|----------|--------|
| **Principal Data Scientist (PDS)** | **yj** | Full-Time | GNN, NLP, DL, ML | Colab A100/V100, Local 4090 | Core models, embeddings, GNN, homology, ensembling, tuning |
| **Data Analyst & Feature Engineer (DAFE)** | **bk** | Part-Time | ML, DL | Local/GCP CPU, Colab | Training assistance, evaluation, submission logistics |

---

# 6. Project Phasing (4 Sprints)

## **Sprint 1 (Weeks 1–2): Feature Launch & Baseline**
**Goal:** Run long compute jobs early, establish a working baseline.

| Task ID | Task | Owner | Deliverable | Dep | Effort | Notes |
|--------|--------|--------|------------|------|--------|-------|
| P1.1 | ESM2-3B Embeddings | yj | `embeddings_3B.h5` | None | 1.5 pd | High GPU job |
| P1.2 | GO DAG & Adjacency Construction | yj | `GO_adj.pkl` | None | 2.0 pd | Needed for GNN |
| P1.3 | Similarity-Aware Split (CD-HIT) | yj | `train_ids_50.csv`, `val_ids_50.csv` | None | 1.0 pd | Prevents leakage |
| P1.4 | Baseline MLP + Fmax Pipeline | yj/bk | `baseline.py` | P1.1, P1.3 | 2.5 pd | First submission |

---

## **Sprint 2 (Weeks 3–4): GNN & Homology Integration**
**Goal:** Develop ontology-aware modeling + launch homology job.

| Task ID | Task | Owner | Deliverable | Dep | Effort | Notes |
|--------|--------|--------|------------|------|--------|-------|
| P2.1 | GNN Model | yj | `GNN_model_v1.pt` | P1.1, P1.2, P1.4 | 4.0 pd | Key innovation |
| P2.2 | MMseqs2 Homology Search | yj | CPU job running | P1.3 | 1.0 pd | Very important signal |
| P2.3 | GNN Review & Optimization | yj/bk | hyperparam notes | P2.1 | 1.0 pd | Joint review |
| P2.4 | Prediction Aggregation Framework | yj | merge code | P1.4, P2.1 | 1.0 pd | Prepares ensemble |

---

## **Sprint 3 (Weeks 5–6): Model Diversity & Stacking**
**Goal:** Build attention model + integrate homology features + ensemble.

| Task ID | Task | Owner | Deliverable | Dep | Effort | Notes |
|--------|--------|--------|------------|------|--------|-------|
| P3.1 | Attention-Pooling Model | yj | `Attn_model_v1.pt` | P1.1, P1.4 | 3.0 pd | Better residue focus |
| P3.2 | Homology Feature Vector Gen | yj | `homology_features.npy` | P2.2 | 3.0 pd | Converts MMseqs2 output |
| P3.3 | Ensemble / Stacking Model | yj | `ensemble_meta_model.pt` | P2.1, P3.1, P3.2 | 4.0 pd | Final scorer |

---

## **Sprint 4 (Weeks 7–8): Tuning, Submission, Finalization**
**Goal:** Final optimization, formatting, and documentation.

| Task ID | Task | Owner | Deliverable | Dep | Effort | Notes |
|--------|--------|--------|------------|------|--------|-------|
| P4.1 | Threshold Optimization | yj | τ\_opt | P3.3 | 2.0 pd | Maximizes Fmax |
| P4.2 | GO Hierarchy Propagation | yj | hierarchy code | P4.1 | 1.0 pd | Ensures parent terms appear |
| P4.3 | Final Submission & Report | bk | `predictions.tsv`, draft report | P4.2 | 3.0 pd | Submission prep |
| P4.4 | Code Cleanup & Repo Review | yj/bk | cleaned repo | P4.2, P4.3 | 1.0 pd | Final polish |

---

# 7. Workload Type Classification

| Type | Description | Tasks |
|------|-------------|--------|
| **HC — Heavy Compute** | Requires GPU/long CPU job | ESM2 embedding, CD-HIT, MMseqs2 |
| **CS — Coding Setup** | Active development | GNN, attention, ensemble |
| **LW — Light Work** | Monitoring/procedural | Documentation, threshold tuning, formatting |

---

# 8. Deliverables Summary

- **Embeddings:** `embeddings_3B.h5`  
- **GO Graph:** `GO_adj.pkl`, GO index  
- **Train/Val Split:** `train_ids_50.csv`, `val_ids_50.csv`  
- **Models:** MLP, GNN, Attention, Ensemble  
- **Features:** Homology vectors  
- **Final Output:** `predictions.tsv` (CAFA6 compliant)

---

# Current Phase

 **Step 1**:  
Set up repository → load FASTA/GO data → run CD-HIT → extract ESM2 embeddings.

