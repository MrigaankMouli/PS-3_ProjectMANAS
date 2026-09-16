# Few-Shot Satellite Object Retrieval Using SwinIR, DINOv2, and Contrastive Patch Matching

## Paper and Reproducibility Guide

This repository supports a journal paper on **query-by-example visual search in low-resolution satellite imagery**. The proposed system accepts one to five example image patches, searches larger satellite scenes, and returns the locations and similarity scores of visually matching objects.

The paper is intentionally limited to the following pipeline:

```text
Query and target imagery
        ↓
Patch extraction and candidate generation
        ↓
SwinIR super-resolution
        ↓
DINOv2 feature extraction
        ↓
Contrastive representation learning
        ↓
Embedding-based patch matching
        ↓
Coordinate recovery, thresholding, and duplicate suppression
        ↓
Ranked object locations and similarity scores
```

Object detectors such as YOLO are outside the scope of this paper.

> **Status:** This README is a manuscript blueprint and reproducibility checklist. Numerical placeholders must be replaced with results from controlled experiments. No accuracy or mAP value should be inferred from the existing prediction CSV alone.

---

## 1. Proposed Paper Title

**Few-Shot Object Retrieval in Low-Resolution Satellite Imagery Using SwinIR-Enhanced DINOv2 Representations and Contrastive Patch Matching**

Alternative titles:

1. **Contrastive Patch Matching for Query-by-Example Search in Low-Resolution Satellite Imagery**
2. **Super-Resolved Foundation-Model Features for Few-Shot Satellite Image Retrieval and Localization**
3. **A SwinIR–DINOv2 Framework for Generic Visual Search in Remote-Sensing Imagery**

---

## 2. Paper Scope

### 2.1 Problem

Large satellite-image collections are difficult to annotate manually. Conventional object detectors require a fixed class vocabulary and substantial labeled training data. In contrast, the target system should locate an object or land feature from only one to five user-supplied examples, including examples from categories not encountered during representation training.

Given a set of query patches

$$
Q = \{q_1,q_2,\ldots,q_m\}, \qquad 1 \leq m \leq 5,
$$

and a target image $I$, the system generates candidate patches

$$
P(I)=\{p_1,p_2,\ldots,p_n\}
$$

and returns a ranked set of matches:

$$
\mathcal{D} = \{(x_{\min},y_{\min},x_{\max},y_{\max},c,s)\},
$$

where $c$ is the query class or user-provided object name and $s$ is the learned similarity score.

### 2.2 Central hypothesis

> SwinIR enhancement followed by contrastive adaptation of DINOv2 embeddings improves few-shot patch retrieval and localization in low-resolution satellite imagery compared with interpolation, frozen DINOv2 features, and non-adapted similarity matching.

### 2.3 Research questions

- **RQ1:** Does SwinIR improve downstream retrieval and localization performance?
- **RQ2:** Does contrastive adaptation make DINOv2 features more discriminative for satellite patches?
- **RQ3:** How does performance change when the number of query examples increases from one to five?
- **RQ4:** Does the method generalize to object categories excluded from contrastive training?
- **RQ5:** How do patch size, scale, stride, and similarity threshold affect accuracy and computational cost?

### 2.4 Intended contributions

The final paper should claim only contributions supported by experiments. Candidate contributions are:

1. An end-to-end query-by-example framework for retrieving and localizing arbitrary features in low-resolution satellite scenes.
2. A task-oriented integration of SwinIR enhancement with DINOv2 representation learning.
3. Contrastive adaptation of foundation-model embeddings for remote-sensing patch matching.
4. A systematic evaluation of few-shot retrieval, localization, unseen-class generalization, and efficiency.
5. An ablation study that isolates the effects of enhancement, representation adaptation, patch generation, and multi-query aggregation.

---

## 3. Dataset Context

The challenge specification is available in [`PS3.pdf`](PS3.pdf). Stage 1 describes:

- Satellite imagery with a spatial resolution of 3 m.
- TIFF files containing blue, green, red, and near-infrared bands.
- Imagery without georeferencing.
- Up to five query chips for each search.
- Target output as bounding-box coordinates, class name, source-image name, and similarity score.
- Reference categories such as playground, brick kiln, metro shed, dried pond, filled pond, sheds, solar panel, and sewage-treatment plant.

The stated challenge partitions include 150 training images, 40 testing images, 40 shortlisting images, nine sample images with annotations, and a separately administered holdout set.

### 3.1 Spectral-band disclosure

Standard pretrained SwinIR and DINOv2 models accept three-channel inputs. The current TIFF visualization code in [`tif_view.py`](tif_view.py) constructs an RGB composite from bands 3, 2, and 1. If the retrieval pipeline follows the same procedure, the paper must describe its input as:

> RGB composites derived from four-band multispectral source imagery.

The paper must not claim four-band learning unless the NIR channel is explicitly processed by the model. If NIR is used, document the fusion strategy, initialization of additional input weights, normalization, and corresponding ablation.

### 3.2 Data-splitting requirements

- Split data by parent satellite scene, not by extracted patch.
- Do not place overlapping patches from the same scene in training and test sets.
- For unseen-class evaluation, exclude every patch of the held-out classes from contrastive training.
- Record the random seed and the exact scene identifiers assigned to each split.
- Report the number of scenes, query patches, positive candidates, negative candidates, and object instances per class.
- Normalize inconsistent labels such as `Shed` and `Sheds` before evaluation.

### 3.3 Dataset table for the manuscript

| Partition | Scenes | Query patches | Positive instances | Negative patches | Classes | Purpose |
|---|---:|---:|---:|---:|---:|---|
| Training | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | Contrastive learning |
| Validation | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | Hyperparameter selection |
| Known-class test | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | In-domain evaluation |
| Unseen-class test | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | Generic-search evaluation |

---

## 4. Proposed Method

### 4.1 Pipeline overview

The query and target branches share the same preprocessing, enhancement, encoder, and projection network. This avoids introducing a representation mismatch between queries and candidate patches.

```text
Query patch q ──→ SwinIR ──→ DINOv2 ──→ projection head ──→ query embedding zq
                                                                          │
                                                                          ├─ cosine similarity
                                                                          │
Target image ─→ candidate patches ─→ SwinIR ─→ DINOv2 ─→ projection ─→ zp
                                                                          │
                                             ranking ← thresholding ←──────┘
                                                 ↓
                              coordinate mapping and duplicate suppression
```

### 4.2 Patch extraction and candidate generation

#### Query patches

The user supplies a query in one of two ways:

1. Uploading an existing image chip.
2. Drawing a bounding box over a reference satellite image.

The selected region is cropped, normalized, and passed through the same processing branch used for target candidates. The implementation must record whether aspect ratio is preserved, padded, or resized directly.

#### Candidate patches

Target scenes are searched using overlapping patches. The manuscript must specify:

- Patch dimensions in original-image pixels.
- Window stride and overlap percentage.
- Number of image scales.
- Aspect-ratio handling.
- Boundary padding.
- Whether candidate sizes depend on query size.
- Number of candidate patches generated per megapixel.

A multiscale candidate set can be represented as:

$$
P(I)=\bigcup_{a\in\mathcal{A}}\operatorname{Slide}(I,w_a,h_a,t_a),
$$

where $(w_a,h_a)$ is the window size and $t_a$ is the stride at scale $a$.

Dense sliding windows maximize coverage but increase inference cost and false positives. The experiments should therefore measure the accuracy–latency trade-off for multiple strides.

### 4.3 SwinIR super-resolution

Each query and candidate patch is enhanced using SwinIR:

$$
\tilde p=S_{\psi}(p),
$$

where $S_{\psi}$ denotes the selected pretrained or fine-tuned SwinIR model. The repository contains a real-image $4\times$ checkpoint in [`models/swinir_real_sr_x4.pth`](models/swinir_real_sr_x4.pth) and supporting code under [`SwinIR/`](SwinIR/).

The paper must report:

- SwinIR variant and checkpoint provenance.
- Upscaling factor.
- Whether parameters are frozen or fine-tuned.
- Input and output normalization.
- Padding or tiling strategy.
- Color-band ordering.
- Processing time per patch.

If an original coordinate is measured after $r\times$ enhancement, it must be mapped back to the source image:

$$
(x,y,w,h)_{\text{source}}=\frac{1}{r}(x,y,w,h)_{\text{enhanced}}.
$$

Because a true high-resolution reference may not exist, SwinIR should be evaluated primarily by its effect on retrieval and localization. Visual sharpness by itself is not evidence of improved scientific fidelity.

### 4.4 DINOv2 feature extraction

The enhanced patch is encoded by DINOv2:

$$
h_i=f_{\theta}(\tilde p_i).
$$

The feature is L2-normalized:

$$
\bar h_i=\frac{h_i}{\lVert h_i\rVert_2}.
$$

The manuscript must identify:

- DINOv2 architecture and model size.
- Generic or remote-sensing-pretrained checkpoint.
- Input dimensions.
- Transformer patch size.
- Output layer.
- CLS-token, mean-token, or attention-pooling strategy.
- Embedding dimension.
- Frozen, partially fine-tuned, or fully fine-tuned backbone.

The repository includes remote-sensing DINOv2 resources under [`dinov2-remote-sensing/`](dinov2-remote-sensing/). Results copied from its README describe the external pretrained model and must not be presented as results of this pipeline.

### 4.5 Contrastive representation learning

DINOv2's original self-supervised training and the pipeline's task-specific contrastive stage are separate processes. The paper must clearly distinguish them.

A projection head maps the DINOv2 representation into the retrieval space:

$$
z_i=\frac{g_{\phi}(\bar h_i)}{\lVert g_{\phi}(\bar h_i)\rVert_2}.
$$

If InfoNCE is used, the per-anchor objective is:

$$
\mathcal{L}_i=-\log
\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}
{\sum_{j\neq i}\exp(\operatorname{sim}(z_i,z_j)/\tau)},
$$

where $z_i^+$ is a positive example and $\tau$ is the temperature.

The implementation and manuscript must document:

- Exact loss function.
- Positive-pair definition.
- Negative-pair definition.
- Hard-negative mining.
- Data augmentations.
- Projection-head architecture.
- Temperature or margin.
- Optimizer and learning-rate schedule.
- Batch size and epochs.
- Backbone freezing policy.
- Checkpoint-selection criterion.
- Random seeds.

Useful hard negatives include visually similar categories such as dried versus filled ponds, metro sheds versus other sheds, and solar arrays versus bright roofs.

### 4.6 Multiple-query aggregation

For $m$ query examples, a normalized mean prototype can be constructed as:

$$
z_Q=\frac{1}{m}\sum_{i=1}^{m}z_{q_i},
\qquad
\hat z_Q=\frac{z_Q}{\lVert z_Q\rVert_2}.
$$

The prototype strategy should be compared with:

- Maximum similarity to any query.
- Mean similarity across queries.
- Medoid selection.
- Learned or similarity-weighted aggregation.

### 4.7 Patch matching

Query-to-candidate similarity is computed using cosine similarity:

$$
s(Q,p)=\hat z_Q^\top z_p.
$$

Candidates are ranked by decreasing similarity. A candidate becomes a proposed match when:

$$
s(Q,p)\geq\delta,
$$

where $\delta$ is selected on the validation set rather than on the test set.

The implementation must state:

- Exhaustive or approximate nearest-neighbor search.
- Number of returned neighbors.
- Similarity threshold.
- Score calibration method.
- Coordinate mapping.
- Non-maximum suppression threshold.
- Duplicate handling across overlapping windows and scales.
- Behavior when no candidate passes the threshold.

### 4.8 Output format

Each accepted match should contain:

```text
x_min,y_min,x_max,y_max,searched_object_name,target_imagery_file_name,similarity_score
```

An existing example is available at [`SiameseNetwork/GC_PS03_31-Oct-2025_AIGR-S66555.csv`](SiameseNetwork/GC_PS03_31-Oct-2025_AIGR-S66555.csv).

---

## 5. Experimental Design

### 5.1 Evaluation protocols

#### Protocol A: Known-class retrieval

Training and test sets may share semantic categories but must contain different parent scenes. This measures in-domain retrieval without patch leakage.

#### Protocol B: Unseen-class retrieval

One or more complete categories are excluded from contrastive training and used only at evaluation time. This is the principal test of the claim that the system supports generic query-by-example search.

#### Protocol C: Few-shot sensitivity

Evaluate independently with one, two, three, and five query examples. Repeat the sampling of query examples and report variability.

#### Protocol D: Cross-condition robustness

Where data permits, evaluate changes in orientation, scale, background, illumination, spatial resolution, or sensor characteristics.

### 5.2 Baselines

The minimum baseline matrix is:

| ID | Input processing | Encoder | Adaptation | Purpose |
|---|---|---|---|---|
| B1 | Original RGB | Pixel or histogram descriptor | None | Classical baseline |
| B2 | Original RGB | Frozen DINOv2 | None | Foundation-model baseline |
| B3 | Bicubic $4\times$ | Frozen DINOv2 | None | Controls for resizing |
| B4 | SwinIR $4\times$ | Frozen DINOv2 | None | Isolates SwinIR |
| B5 | Original RGB | DINOv2 | Contrastive | Isolates contrastive adaptation |
| Ours | SwinIR $4\times$ | DINOv2 | Contrastive | Complete pipeline |

Optional baselines include SIFT, color histograms, ResNet embeddings, a Siamese CNN, a remote-sensing-pretrained encoder, or a CLIP-family model.

### 5.3 Metrics

#### Retrieval metrics

- Precision@$K$
- Recall@$K$
- Average Precision (AP)
- Mean Average Precision (mAP)
- Normalized Discounted Cumulative Gain (nDCG), if ranking quality is central

#### Localization metrics

For predicted box $B_p$ and ground-truth box $B_g$:

$$
\operatorname{IoU}(B_p,B_g)=
\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
$$

Report:

- Precision, recall, and F1 at a stated IoU threshold.
- AP per category.
- mAP@0.5.
- mAP@0.5:0.95, if annotation volume supports it.

#### Efficiency metrics

- Candidate patches per second.
- End-to-end time per target image.
- Processing time per megapixel.
- Query latency after index creation.
- Peak CPU/GPU memory.
- Embedding-index storage.

### 5.4 Ablation studies

The following experiments are needed to justify the design:

1. SwinIR versus no enhancement.
2. SwinIR versus bicubic interpolation.
3. Different enhancement factors.
4. Frozen versus contrastively adapted DINOv2.
5. Alternative DINOv2 layers and pooling strategies.
6. Projection-head depth and embedding dimension.
7. Contrastive-loss temperature or margin.
8. Random versus hard negatives.
9. Patch size and stride.
10. Single-scale versus multiscale candidates.
11. One, two, three, and five query examples.
12. Query-aggregation strategy.
13. Similarity threshold and NMS threshold.
14. RGB versus RGB+NIR, only if NIR support is implemented.

### 5.5 Statistical analysis

- Repeat experiments with at least three random seeds where training is stochastic.
- Repeat few-shot trials with multiple query selections.
- Report mean and standard deviation.
- Provide bootstrap confidence intervals for principal metrics.
- Use paired tests or paired bootstrap comparisons for major pipeline variants.
- Report per-class results so that aggregate gains are not driven by one frequent category.

---

## 6. Results Template

No values in this section should be completed until evaluation against ground-truth annotations is available.

### 6.1 Main comparison

| Method | Recall@1 | Recall@5 | Retrieval mAP | Localization mAP@0.5 | F1 | Time/image |
|---|---:|---:|---:|---:|---:|---:|
| Original RGB + frozen DINOv2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Bicubic + frozen DINOv2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| SwinIR + frozen DINOv2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| RGB + contrastive DINOv2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Full pipeline | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 6.2 Per-class localization

| Class | Ground-truth objects | AP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Playground | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Brick kiln | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Metro shed | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Pond-1 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Pond-2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Sheds | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Solar panel | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| STP | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 6.3 Known versus unseen classes

| Evaluation condition | Classes | Recall@5 | Retrieval mAP | Localization mAP@0.5 |
|---|---:|---:|---:|---:|
| Known classes | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Unseen classes | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 6.4 Number of examples

| Query examples | Recall@1 | Recall@5 | Localization mAP@0.5 |
|---:|---:|---:|---:|
| 1 | `[TBD]` | `[TBD]` | `[TBD]` |
| 2 | `[TBD]` | `[TBD]` | `[TBD]` |
| 3 | `[TBD]` | `[TBD]` | `[TBD]` |
| 5 | `[TBD]` | `[TBD]` | `[TBD]` |

### 6.5 Efficiency

| Configuration | Patches/image | Time/image | Peak memory | Index size | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| Small stride | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Medium stride | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Large stride | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 7. Journal Manuscript Structure

### Abstract

Write a 200–250-word abstract containing:

1. The annotation and generic-search problem.
2. The one-to-five-example setting.
3. The SwinIR–DINOv2–contrastive matching method.
4. The dataset and evaluation protocols.
5. The principal quantitative findings.
6. The conclusion and practical implication.

Suggested abstract skeleton:

> Searching large satellite-image archives for user-defined objects is challenging because supervised detectors require extensive class-specific annotations and small objects are poorly represented in low-resolution imagery. This paper proposes `[METHOD NAME]`, a few-shot query-by-example framework that accepts one to five reference patches and localizes visually corresponding regions in satellite scenes. Candidate patches are enhanced using SwinIR, encoded using DINOv2, and mapped into a task-specific embedding space through contrastive learning. Candidate locations are ranked by cosine similarity and consolidated through thresholding and non-maximum suppression. The approach is evaluated on `[DATASET]` using known-class, unseen-class, and few-shot protocols. Compared with `[STRONGEST BASELINE]`, the proposed method improves `[PRIMARY METRIC]` by `[VALUE]` while requiring `[RUNTIME]`. Ablation experiments show that `[SUPPORTED FINDING]`. These findings demonstrate `[CONCLUSION]`.

### Keywords

`content-based image retrieval`; `remote sensing`; `few-shot learning`; `contrastive learning`; `DINOv2`; `SwinIR`; `satellite imagery`; `visual search`; `image matching`

### 1. Introduction

1. Explain the growth of satellite-image archives and cost of expert annotation.
2. Explain why fixed-class object detection does not solve arbitrary query-by-example search.
3. Identify low spatial resolution, small targets, scale variation, and complex background as key challenges.
4. Introduce the proposed pipeline at a conceptual level.
5. State the research gap and research questions.
6. End with a numbered contribution list and a short description of the remaining sections.

### 2. Related Work

#### 2.1 Remote-sensing content-based image retrieval

Review handcrafted descriptors, CNN retrieval systems, transformer features, and the difference between whole-scene retrieval and object-level localization.

#### 2.2 Few-shot and query-by-example localization

Review Siamese networks, prototypical methods, metric learning, and sliding-window matching. Explain why the current method is retrieval-based rather than a conventional detector.

#### 2.3 Image super-resolution in remote sensing

Review CNN and transformer super-resolution, including SwinIR. Emphasize that visually plausible detail may not correspond to genuine spatial information.

#### 2.4 Self-supervised vision transformers

Review DINO and DINOv2, token representations, transfer learning, and remote-sensing adaptation.

#### 2.5 Contrastive representation learning

Review contrastive loss, triplet loss, InfoNCE, supervised contrastive learning, hard-negative mining, and embedding-based retrieval.

End the section by identifying the unresolved problem addressed by the paper.

### 3. Materials and Methods

Recommended subsections:

1. Problem formulation.
2. Dataset and preprocessing.
3. Patch generation.
4. SwinIR enhancement.
5. DINOv2 representation extraction.
6. Contrastive adaptation.
7. Multi-query aggregation.
8. Similarity matching and localization.
9. Implementation details.

### 4. Experimental Setup

Recommended subsections:

1. Data partitions and leakage prevention.
2. Known-class protocol.
3. Unseen-class protocol.
4. Few-shot protocol.
5. Baselines.
6. Metrics.
7. Hyperparameters.
8. Statistical analysis.

### 5. Results

Recommended subsections:

1. Overall retrieval results.
2. Localization results.
3. Known- versus unseen-class generalization.
4. Effect of query-set size.
5. Ablation study.
6. Efficiency analysis.
7. Qualitative successes and failure cases.

### 6. Discussion

Discuss:

- Whether SwinIR improves matching rather than only apparent sharpness.
- Why contrastive adaptation changes retrieval performance.
- Which categories benefit most and least.
- The role of object size and background context.
- Failure modes caused by scale, orientation, or appearance changes.
- Generalization to unseen categories.
- Accuracy–latency trade-offs.
- Risks associated with super-resolution hallucination.

Do not repeat the results table. Interpret the evidence and connect it to the research questions.

### 7. Limitations

At minimum, address:

- Limited annotated imagery.
- Any exclusion of the NIR band.
- Dependence on chosen patch scales and stride.
- Dense-search computational cost.
- Potential SwinIR hallucination.
- Sensitivity to similarity and NMS thresholds.
- Lack of georeferencing in the present data.
- Rectangular localization of irregular objects.
- Cross-sensor and cross-resolution generalization.
- Missing or incomplete implementation artifacts.

### 8. Conclusion

Restate the problem, summarize the method, answer the research questions using measured evidence, and identify future work. Do not introduce a new experiment or result in the conclusion.

### Declarations

Include journal-required statements:

- Data availability.
- Code availability.
- Funding.
- Conflict of interest.
- Author contributions.
- Acknowledgments.
- Ethical statement, if required.

---

## 8. Required Figures

1. **System overview:** Complete query and target branches.
2. **Patch generation:** Multiscale sliding-window or candidate-selection process.
3. **Enhancement examples:** Original, bicubic, and SwinIR patches at the same displayed scale.
4. **Contrastive training:** Anchor, positive, easy negative, and hard negative construction.
5. **Embedding visualization:** UMAP or t-SNE before and after contrastive adaptation.
6. **Precision–recall curves:** Overall and selected difficult categories.
7. **Few-shot curve:** Performance against number of query examples.
8. **Efficiency curve:** Accuracy against stride, candidates, or latency.
9. **Qualitative results:** Correct matches, false positives, false negatives, and SwinIR failure cases.

Every qualitative figure should show query patch, target scene, predicted box, ground-truth box, similarity score, and class label.

---

## 9. Required Tables

1. Dataset composition and splits.
2. Model architecture and hyperparameters.
3. Main baseline comparison.
4. Per-class retrieval and localization results.
5. Component ablation.
6. Patch-size and stride ablation.
7. One-to-five-query comparison.
8. Known- versus unseen-class results.
9. Runtime, memory, and storage comparison.

---

## 10. Reproducibility Checklist

Before submission, verify that the repository records all of the following:

- [ ] Exact scene-level train, validation, and test splits.
- [ ] Class-name normalization rules.
- [ ] Query-generation procedure.
- [ ] Patch sizes, strides, scales, and padding.
- [ ] RGB/NIR handling and band order.
- [ ] SwinIR architecture, weights, and scale.
- [ ] DINOv2 architecture and checkpoint.
- [ ] DINOv2 layer and token-pooling method.
- [ ] Projection-head architecture.
- [ ] Contrastive objective and pair-sampling algorithm.
- [ ] Hard-negative mining procedure.
- [ ] Augmentations.
- [ ] Training hyperparameters and random seeds.
- [ ] Query-aggregation rule.
- [ ] Similarity threshold-selection procedure.
- [ ] Coordinate conversion and NMS implementation.
- [ ] Evaluation code and IoU convention.
- [ ] Hardware and software versions.
- [ ] Trained checkpoint hashes.
- [ ] Per-run raw metrics.
- [ ] Failure-case examples.

---

## 11. Current Repository Evidence and Gaps

### Available artifacts

- The formal challenge description: [`PS3.pdf`](PS3.pdf).
- Four-band TIFF imagery and JSON reference annotations under [`PS_3/Datasets/`](PS_3/Datasets/).
- TIFF/RGB visualization logic: [`tif_view.py`](tif_view.py).
- SwinIR source and test resources: [`SwinIR/`](SwinIR/).
- A real-image $4\times$ SwinIR checkpoint: [`models/swinir_real_sr_x4.pth`](models/swinir_real_sr_x4.pth).
- A partial SwinIR notebook: [`swinir.ipynb`](swinir.ipynb).
- Remote-sensing DINOv2 resources: [`dinov2-remote-sensing/`](dinov2-remote-sensing/).
- A shortlisting prediction file: [`SiameseNetwork/GC_PS03_31-Oct-2025_AIGR-S66555.csv`](SiameseNetwork/GC_PS03_31-Oct-2025_AIGR-S66555.csv).

### Gaps that must be resolved before paper submission

- [`DinoV2/image_retrieval.ipynb`](DinoV2/image_retrieval.ipynb) is currently empty.
- [`swinir.ipynb`](swinir.ipynb) contains only partial loading and preprocessing logic.
- The submitted archive under [`SiameseNetwork/`](SiameseNetwork/) contains output artifacts but no contrastive-training implementation.
- The exact positive/negative sampling strategy is not documented.
- The exact contrastive loss and projection head are not documented.
- The patch extraction, coordinate recovery, thresholding, and NMS code is not present as a unified reproducible pipeline.
- Ground-truth evaluation metrics are not stored in the repository.
- The existing prediction CSV provides candidate boxes and scores but cannot by itself establish mAP, precision, recall, or accuracy.
- Label names require normalization, including the `Shed`/`Sheds` inconsistency.

These gaps do not prevent drafting the paper, but they prevent reproducible experimental claims until addressed.

---

## 12. Recommended Project Organization

The following structure would make the implementation consistent with the manuscript:

```text
PS-3/
├── README.md
├── PS3.pdf
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   └── retrieval.yaml
├── data/
│   ├── raw/
│   ├── annotations/
│   └── splits/
├── src/
│   ├── patches.py
│   ├── swinir_enhancer.py
│   ├── dinov2_encoder.py
│   ├── contrastive_model.py
│   ├── train_contrastive.py
│   ├── build_index.py
│   ├── search.py
│   ├── postprocess.py
│   └── evaluate.py
├── checkpoints/
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
└── tests/
```

Large datasets and model weights should not be duplicated. Configuration files should reference their locations.

---

## 13. Claims to Avoid Without Evidence

Do not write any of the following unless the corresponding experiment exists:

- “SwinIR recovers lost ground-truth detail.”
- “The system processes all four spectral bands.”
- “The method is class agnostic.”
- “The method generalizes to unseen objects.”
- “The complete pipeline outperforms DINOv2.”
- “The system achieves state-of-the-art performance.”
- “Similarity scores are calibrated probabilities.”
- “The method is real time.”

Use narrower language such as “enhances the input patch,” “produces a ranked similarity score,” or “was evaluated on held-out categories” when that wording matches the evidence.

---

## 14. Immediate Next Steps

1. Consolidate patch extraction, enhancement, encoding, training, matching, and evaluation into reproducible scripts.
2. Freeze a scene-level dataset split and save it as text or JSON.
3. Normalize category names.
4. Establish the frozen-DINOv2 and bicubic baselines.
5. Train the contrastive projection head with documented positive and negative sampling.
6. Run the full ablation matrix.
7. Evaluate known and unseen categories separately.
8. Save raw predictions and per-run metrics.
9. Generate the required figures and tables.
10. Replace every `[TBD]` value in this README and the manuscript with traceable experimental results.

---

## 15. Suggested Paper Narrative in One Paragraph

This study addresses query-by-example localization of user-defined objects in low-resolution satellite imagery without training a fixed-class detector. Query and target images are decomposed into comparable patches, enhanced using SwinIR, and encoded using DINOv2. A task-specific contrastive objective reorganizes the pretrained feature space so that semantically corresponding satellite patches are close while visually confusing background or nonmatching patches are separated. One to five query embeddings are aggregated into a search representation, candidate patches are ranked using cosine similarity, and high-scoring overlapping candidates are converted to image coordinates and consolidated. The central evaluation determines whether enhancement and contrastive adaptation improve retrieval, localization, and unseen-class generalization relative to frozen DINOv2 and interpolation-based baselines, while also measuring the computational cost of dense patch search.

