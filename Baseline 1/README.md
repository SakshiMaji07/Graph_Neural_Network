# GNN-based Hypothesis Selection for SGMH-SLAM

This project studies a graph-based hypothesis selection problem for Simultaneous Localization and Mapping (SLAM). In the setting considered here, the system is given a single ground-truth structural graph and five candidate SLAM hypothesis graphs. The goal is to score each hypothesis against the ground truth and select the one that is most structurally consistent with the true map state. The core idea is to learn a graph similarity function with a shared Graph Neural Network (GNN) encoder, rather than relying only on hand-designed graph distances.

## Problem Statement

We consider a multi-hypothesis SLAM setting in which the estimator produces several candidate scene interpretations, each represented as a graph. One graph corresponds to the ground-truth structural layout of the environment, while the remaining five graphs are alternative hypotheses produced during noisy or ambiguous inference.

The task is:

- receive one ground-truth structural graph;
- receive five candidate SLAM hypothesis graphs;
- compute a similarity score for each hypothesis relative to the ground truth;
- choose the hypothesis whose graph is most similar to the ground-truth graph.

Formally, if $G_{gt}$ denotes the ground-truth graph and $\{G_i\}_{i=1}^{5}$ denotes the set of five candidate hypothesis graphs, the objective is to identify

$$
\hat{i} = \arg\max_{i \in \{1,\dots,5\}} S(G_{gt}, G_i),
$$

where $S(\cdot, \cdot)$ is a learned or engineered graph similarity function that assigns a continuous value in $[0, 1]$.

This framing is useful because SLAM hypothesis selection is a decision problem under uncertainty: multiple map explanations may be locally plausible, but only one should be selected when the graph structure best matches the true environment.

## Baseline Architecture

The baseline model is designed to be simple, modular, and interpretable while still capturing relational structure in the graphs.

```text
Hypothesis graph
        ↓
Shared GNN encoder
        ↓
Graph embedding

Ground truth graph
        ↓
Same GNN encoder
        ↓
Graph embedding

Embeddings
        ↓
difference/product comparison
        ↓
MLP
        ↓
continuous similarity [0,1]
```

### Overview

Each graph is passed through a shared encoder, producing a latent representation of the graph. The same encoder is applied to both the candidate hypothesis graph and the ground-truth graph, ensuring that both graphs are mapped into a comparable embedding space.

The two graph embeddings are then combined using a comparison operator such as:

- element-wise difference,
- element-wise product,
- or a concatenation of both, depending on the specific implementation.

This comparison vector is then fed into a small multi-layer perceptron (MLP), which produces a scalar similarity prediction in the continuous range $[0,1]$. The output is interpreted as the degree of match between the hypothesis graph and the ground-truth graph.

### Design Principles

The baseline intentionally follows a compact design:

- a shared encoder enforces consistency across graphs;
- a learned comparison head captures nontrivial structural alignment;
- the final output is a continuous score suitable for ranking hypotheses.

This baseline serves as a reference point for evaluating whether more advanced structural or matching mechanisms improve hypothesis discrimination.

## Training

Each hypothesis-ground-truth pair forms a training example. For a given scene, one graph acts as the true structural graph and another graph is a candidate hypothesis. The model predicts a score that reflects how similar the two are. During training, the objective is to maximize the similarity score for the hypothesis that most closely matches the ground truth and to penalize poor matches.

This gives rise to a supervised pairwise training setup:

- one positive or strongly compatible pair (ground truth with best match);
- one or more negative or weaker matches (other hypotheses);
- the model learns to assign higher scores to more accurate graph matches.

The training pipeline is therefore framed as graph pair classification/regression: given a graph pair $(G_{gt}, G_h)$, predict a score $s \in [0,1]$ that estimates their similarity.

The training data is naturally organized as graph pairs rather than isolated graphs, which makes the problem amenable to pairwise learning and ranking-based objective functions.

## Inference

At inference time, the model evaluates all five candidate hypotheses independently. Each hypothesis graph is compared against the same ground-truth graph via the shared encoder and scoring head.

The final decision rule is:

$$
\hat{h} = \arg\max_{h \in \{1,\dots,5\}} S(G_{gt}, G_h).
$$

In other words, the system computes one similarity score per hypothesis and selects the argmax as the final chosen hypothesis.

This is important because the task is not a single binary classification problem; it is a five-way hypothesis selection problem in which the model must rank the candidates according to structural agreement with the true graph.

## Optional Extensions

The repository includes a set of optional extensions designed to improve the baseline when higher fidelity to graph structure is required. These components are separated from the baseline to maintain a clear comparison and to isolate which modeling decisions actually contribute to performance.

### Edge-aware Message Passing

Standard message passing often aggregates information from neighboring nodes using node features alone. In edge-aware message passing, edge attributes are incorporated into the update process so that the model can reason about both node identities and relational structure.

This is relevant because in SLAM graphs, edges often encode geometric, spatial, or connectivity information that is as important as node features. Ignoring edge information may cause the network to underrepresent differences between similar-looking but structurally distinct hypotheses.

### Node-level Graph Matching

Rather than summarizing an entire graph into a single embedding, node-level matching attempts to align nodes between two graphs and assess local correspondences. This is useful when graphs are partially observed, noisy, or contain repeated substructures.

Node-level matching can improve robustness by considering whether individual graph components align rather than only whether the global summary embedding looks similar.

### Affinity Matrix

An affinity matrix represents pairwise compatibility between nodes in two graphs. For a graph pair $(G_1, G_2)$, the affinity matrix $A$ contains scores indicating how compatible each node in $G_1$ is with each node in $G_2$.

This matrix can be computed from node features, edge features, or learned pairwise embeddings. It provides a differentiable bridge between local node similarity and global matching decisions.

### Sinkhorn

Sinkhorn normalization is a differentiable approximation to an optimal transport or assignment problem. It can be applied to the affinity matrix to obtain a soft assignment matrix that preserves row and column normalization constraints.

This is useful for graph matching because it transforms a raw compatibility score matrix into a near-bistochastic matching distribution, enabling smoother learning of correspondences. The Sinkhorn procedure is often used in differentiable matching pipelines where exact assignment is intractable or brittle.

### Supervised Node Correspondence

In some settings, the graph matching process is supervised with explicit node-to-node correspondence labels. This provides stronger training signal than only a global graph similarity score. When node correspondences are available, the model can directly learn to align nodes across graphs.

This is especially useful when partial graph observations or repeated structures make global similarity ambiguous.

### Ranking Loss

Ranking loss encourages the model to assign higher scores to better hypotheses relative to worse ones. In a five-hypothesis selection task, this means the model is trained not only to predict a similarity value for a single pair, but to impose a consistent ordering across multiple candidate hypotheses.

This is relevant for selection tasks because the decision depends on relative ranking, not just absolute calibration. A ranking loss can improve the ability to separate the top candidate from the remaining four options.

## Why These Extensions Are Separated from the Baseline

The optional extensions are intentionally kept separate from the baseline for methodological clarity and ablation control.

This separation matters because it allows the project to answer targeted research questions:

- Does a stronger graph encoder alone improve performance?
- Does edge information materially help discrimination?
- Does node-level matching improve robustness under partial or noisy observations?
- Does Sinkhorn-based matching contribute beyond a simple pooled representation?
- Does ranking loss improve five-way selection quality?

If all extensions were merged into the baseline implementation, it would become difficult to determine which component caused an observed gain or failure. By isolating each component in a modular pipeline, the project supports clean empirical comparison and rigorous ablation studies.

In short, the baseline provides a minimal reference model, while the extensions represent hypothesis-driven improvements that can be tested and compared under controlled conditions.

## Dataset Structure

The project assumes a structured dataset of graph pairs organized by split. The repository contains the following high-level layout:

```text
data/
  raw/
    train/
    val/
    test/
  processed/
    train/
    val/
    test/
  splits/
    train.txt
    val.txt
    test.txt
```

### Data Organization

- `raw/`: original graph data before preprocessing.
- `processed/`: cleaned and normalized graph data ready for model consumption.
- `splits/`: text files listing dataset ids for each split.

A typical example is a graph dataset in which each sample contains a graph representation of a scene or structural map, including node features, edge features, and adjacency information. Each sample may be paired with a ground-truth graph or with one of several candidate hypotheses depending on the task configuration.

### Graph Representation

Graphs are expected to encode:

- nodes: entities or landmarks in the map structure;
- edges: pairwise relationships, connectivity, or geometric constraints;
- node attributes: features describing local structure or measurements;
- edge attributes: information about relation strength, distance, or type.

The training data is pair-oriented: a candidate graph and a ground-truth graph are treated as one example for the similarity network.

## Installation

The repository is designed for a Python-based research environment. Install the required dependencies as follows:

```bash
git clone <repository-url>
cd "GNN SLAM\Baseline 1"
python -m venv .venv
# On Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using a CUDA-enabled environment, ensure that the appropriate PyTorch build is installed for your platform and driver version before running training scripts.

## Training Commands

Training is configured through YAML files and the training entry points in the `scripts/` directory. The project includes configuration files such as:

- `configs/base.yaml`
- `configs/baseline_siamese.yaml`
- `configs/edge_aware.yaml`
- `configs/sinkhorn.yaml`

Typical training commands are structured as follows:

```bash
python scripts/train_baseline.py --config configs/baseline_siamese.yaml
python scripts/train_baseline.py --config configs/edge_aware.yaml
python scripts/train_baseline.py --config configs/sinkhorn.yaml
```

Additional flags may include:

- dataset split selection,
- learning rate and batch size,
- number of epochs,
- checkpoint save path,
- device selection (`cuda` or `cpu`),
- random seed control.

The baseline configuration should be used first to establish a reference. Advanced configurations can then be compared against it under identical training conditions.

## Evaluation

Evaluation focuses on whether the system correctly selects the best hypothesis among the five candidates. The main evaluation metric is the final selection accuracy:

- proportion of examples for which the selected hypothesis matches the true best hypothesis;
- optionally reported per split: train, validation, and test.

In addition, the project may report:

- ranking metrics for the top-k hypotheses,
- calibration of the continuous similarity scores,
- distribution of score margins between the best and second-best hypothesis,
- robustness under graph noise, sparsity, or occlusion.

The evaluation script is intended to load saved model checkpoints, score each hypothesis graph against the ground-truth graph, and apply the final argmax rule.

```bash
python scripts/evaluate.py --checkpoint <path-to-checkpoint> --config <config-file>
```

## Experiment Tracking and Ablations

This project is intended for controlled experimentation. All experiments should be tracked with a reproducible configuration and a clear logging setup.

Recommended artifacts to record for each run:

- config file used,
- random seed,
- dataset split, preprocessing version,
- model checkpoint,
- validation metrics,
- training curves,
- environment dependencies and package versions,
- experiment notes summarizing changes.

### Ablation Design

The repository supports a natural ablation structure by varying one factor at a time:

1. Baseline Siamese GNN
2. Edge-aware message passing
3. Node-level matching abstraction
4. Affinity + Sinkhorn matching
5. Supervised node correspondence
6. Ranking loss versus standard similarity regression

Each ablation should keep the rest of the pipeline fixed unless the experiment explicitly targets interaction effects. This design makes it possible to attribute performance changes to specific modeling choices.

The key objective is not just to maximize a single metric but to understand which modeling assumptions contribute to robust hypothesis selection under structural uncertainty.

## Research Questions

This project is intended to answer the following questions:

- Does learned graph similarity outperform hand-designed graph similarity?
- Does edge information improve hypothesis discrimination?
- Does node-level matching improve robustness to partial observations?
- Does ranking loss improve five-way hypothesis selection?
- How does performance change with graph sparsity or noise?
- Is the learned similarity score better calibrated than simple relational heuristics?
- Does the added complexity of matching-based methods justify their gain over the baseline?

These questions highlight the central goal of the project: understanding when and why structurally aware graph learning improves SLAM hypothesis selection compared with simpler similarity baselines.

## Limitations

Although the baseline and extensions are useful for graph-based hypothesis selection, several limitations should be acknowledged:

- The model depends on the quality and completeness of graph representations extracted from SLAM outputs.
- Learned similarity may degrade under domain shift if graph statistics differ from the training distribution.
- Sparse or noisy graphs can make structural correspondence ambiguous, even for strong matching models.
- A graph embedding may lose fine-grained local details that matter for accurate node-level reasoning.
- Node-level matching methods can be computationally more expensive and less stable than global embedding approaches.
- Ranking-oriented objectives may improve selection quality without improving absolute score calibration.

These limitations motivate careful experimental design, robust preprocessing, and transparent ablation reporting.

## Reproducibility Checklist

To support reproducibility, the following checklist should be followed for every experiment:

- [ ] Preserve the exact config file and training command.
- [ ] Record the random seed used for data splits and model initialization.
- [ ] Log all hyperparameters and optimizer settings.
- [ ] Ensure the dataset split and preprocessing version are fixed.
- [ ] Save the model checkpoint and evaluation outputs.
- [ ] Report validation and test metrics under the same conditions.
- [ ] Document the software environment, including dependency versions.
- [ ] Run ablations with controlled, single-variable changes.
- [ ] Retain the script versions used for data preparation, training, and evaluation.
- [ ] Verify results on multiple random seeds when possible.

## Summary

This project addresses a fundamental problem in multi-hypothesis SLAM: selecting the most accurate candidate map graph among several alternatives. The baseline framework learns a similarity function over graph pairs using a shared GNN encoder and a small MLP head. The optional extensions expand this idea with edge-aware reasoning, node-level matching, affinity-based assignment, Sinkhorn normalization, and ranking objectives.

The central research value of the project lies in understanding which forms of structural reasoning are most effective for hypothesis discrimination, and how far a learned graph similarity model can go beyond hand-designed graph comparison heuristics.
