# SMTE – GutBrainIE@CLEF 2026: Named
Entity Recognition and Mention-Level Relation Extraction for the Gut-Brain Axis
This repository contains the implementation of **SMTE**, our system developed for **Task 6 of the BioASQ CLEF Lab 2026 — GutBrainIE**, a shared task on structured information extraction from PubMed abstracts related to the gut microbiota and its connections with neurological and mental health conditions (Alzheimer's, Parkinson's, Multiple Sclerosis, ALS, and mental health disorders).

The project is part of the EU-supported [HEREDITARY](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) project.

> **Paper:** *SMTE – GutBrainIE@CLEF 2026: Biomedical Named Entity Recognition and Mention-Level Relation Extraction for the Gut-Brain Axis*
> Sofia Maule, Giorgio Maria Di Nunzio — University of Padova, Department of Information Engineering
> CLEF 2026 Working Notes, BioASQ Lab

Two subtasks are addressed:

- **Subtask 6.1.1** — Named Entity Recognition (NER) → **Micro-F1 0.8019 on test set, ranked 5th**
- **Subtask 6.2.1** — Mention-Level Relation Extraction (M-RE) → **Micro-F1 0.4223 on test set, ranked 2nd**

---

## Tasks

### Subtask 6.1.1 — Named Entity Recognition (NER)

Given a PubMed abstract discussing the gut-brain interplay, the system identifies and classifies text spans into one of **13 predefined biomedical categories**: `anatomical location`, `animal`, `bacteria`, `biomedical technique`, `chemical`, `DDF` (disease, disorder, or finding), `dietary supplement`, `drug`, `food`, `gene`, `human`, `microbiome`, `statistical technique`.

Each prediction is expressed as a tuple:
```
(entityCategory ; entityLocation ; startOffset ; endOffset)
```
where `entityLocation` is `title` or `abstract`, and offsets are character-level.

**Submitted runs:**

| Run | System | Dev Mic-F1 | Test Mic-F1 |
|-----|--------|-----------|------------|
| R1 (primary) | NERensemble — hybrid BioBERT + PubMedBERT | 0.8324 | **0.8019** |
| R2 | NERpubmedbert — single PubMedBERT | 0.8270 | 0.7961 |
| R3 | NERbiobert — single BioBERT | 0.8194 | 0.7951 |
| — | Organizer baseline (GLiNER) | — | 0.7996 |

**Training strategies explored:**

| Config | Description |
|--------|-------------|
| NB1 | Single-stage on Gold + Silver (BioBERT) |
| NB1b | Single-stage on Gold + Silver + Bronze (BioBERT / PubMedBERT) — **best single model** |
| NB2 | Two-stage curriculum: Gold first, then Gold + Silver |
| NB3 | Three-stage cross-year: 2025 data → 2026 Gold → 2026 Gold + Silver |
| NB4 | Annotator-weighted training on Gold + Silver + Silver 2025 |
| NB4b | Annotator-weighted training including Bronze |

**Inference pipeline:** BIO decoding with median-token confidence scoring → label-specific two-pass thresholding → label-aware post-processing heuristics (gene/chemical and food/dietary supplement remapping) → deduplication with overlap resolution.

**Ensemble strategy (R1):** PubMedBERT predictions as base; non-overlapping BioBERT spans added for all labels.

---

### Subtask 6.2.1 — Mention-Level Relation Extraction (M-RE)

Given a PubMed abstract, the system identifies relations between entity mention pairs and assigns the correct predicate. The schema defines **17 semantic predicates** (`influence`, `is linked to`, `change abundance`, `impact`, `administered`, `affect`, `change effect`, `change expression`, `compared to`, `interact`, `is a`, `located in`, `part of`, `produced by`, `strike`, `target`, `used by`) over 55 legal type-pair patterns across 52 unique subject–object type combinations.

Each prediction is expressed as a triple:
```
(subjectMention ; relationPredicate ; objectMention)
```

Entity mentions at inference time come from the **NER ensemble (R1 of Subtask 6.1.1)**, making the system a true end-to-end pipeline.

**Submitted runs:**

| Run | System | Dev Mic-F1 | Test Mic-F1 |
|-----|--------|-----------|------------|
| R1 (primary) | REfullctx — PubMedBERT-large, full abstract context | 0.5961 | 0.4190 |
| R2 | RElargewindow — PubMedBERT-large, ±300-char window | 0.5932 | **0.4223** |
| — | Organizer baseline (ATLOP) | — | 0.3886 |

R2 outperformed R1 on the test set despite ranking second on dev, suggesting the full-context model overfits to the development distribution.

**Training configurations explored:**

| Config | Backbone | Pooling | Dev Mic-F1 |
|--------|----------|---------|-----------|
| A0 | BiomedBERT-base | marker-only | 0.5753 |
| A3 | BiomedBERT-base | mention-mean | 0.5780 |
| A4 | BiomedBERT-base | mention-mean + label smoothing | 0.5748 |
| A5 | BiomedBERT-base / PubMedBERT-large | mention-mean + hard negatives | 0.5899 / 0.5932 |
| A5-BioLink | BioLinkBERT-base | mention-mean + hard negatives | 0.5845 |
| **A5-fullctx** | **PubMedBERT-large** | **mention-mean + hard neg + full ctx** | **0.5961** |
| A6-typed | BiomedBERT-base | mention-mean + typed markers | 0.5005 |
| A7 | BiomedBERT-base | mention-mean + typed markers + hard neg | 0.5444 |

**Inference pipeline:** Logit caching → legal-only decoding with temperature scaling (T=1.25) → distance-aware minimum probability floor → per-predicate threshold tuning (margin, min_prob, max_chars, top2_gap) → deduplication.

---

## Dataset

Training data is provided in JSON format, organized into four quality tiers:

| Tier | Description |
|------|-------------|
| **Gold** | Expert-curated annotations |
| **Silver** | Mid-quality annotations by trained linguistics students |
| **Silver 2025** | Annotations from the 2025 edition; concept-level info auto-generated |
| **Bronze** | Distantly supervised, fully automatic (GLiNER for NER, ATLOP for RE) |

The test set is a held-out subset of Gold data. The RE training set contains 4,921 documents (639 Gold, 811 Silver, 499 Silver 2025, 2,972 Bronze), yielding 53,791 positive and 194,662 negative relation examples (ratio 3.6:1).

---

## Repository Structure

```
GutBrainIE/
│
├── data/
│   ├── GutBrainIE_Full_Collection_2025/       # Training data (2025 edition)
│   └── GutBrainIE_Full_Collection_2026/       # Training data (2026 edition)
│   └── SMTE_GutBrainIE_2026                   # final submission runs
│
├── src/
│   ├── ner/                                   # NER subtask (6.1.1)
│   │   ├── bert_NER_NB1_twopass.ipynb         # NB1: Gold+Silver, single-stage
│   │   ├── bert_NER_NB2_twopass_stage_training.ipynb     # NB2: Two-stage curriculum
│   │   ├── bert_NER_NB3_twopass_weight_train2526.ipynb   # NB3: Cross-year curriculum
│   │   ├── bert_NER_NB4_twopass_weighted_annotators.ipynb        # NB4: Annotator weights
│   │   ├── bert_NER_NB4b_twopass_weighted_annotators_bronze.ipynb # NB4b: +Bronze
│   │   ├── bert_NER_ensemble.ipynb            # R1: hybrid BioBERT + PubMedBERT ensemble
│   │   └── bert_NER_inference_all_models.ipynb
│   │
│   ├── re/                                    # RE subtask (6.2.1)
│   │   ├── training/
│   │   │   ├── bert_RE_A0_fixed.ipynb         # A0: marker-only, uniform negatives
│   │   │   ├── bert_RE_A2_weighted_fixed.ipynb  # A2: quality-weighted loss
│   │   │   ├── bert_RE_A3_mentionmean.ipynb   # A3: mention-mean pooling
│   │   │   ├── bert_RE_A4_labelsmooth.ipynb   # A4: label smoothing
│   │   │   ├── bert_RE_A5_hardneg.ipynb       # A5: hard negative sampling
│   │   │   ├── bert_RE_A5_fullctx.ipynb       # R1: full context — primary system
│   │   │   ├── bert_RE_A6_typed.ipynb         # A6: typed entity markers
│   │   │   └── bert_RE_A7_typed_hardneg.ipynb # A7: typed markers + hard negatives
│   │   │
│   │   ├── bert_RE_ensemble.ipynb             # Post-hoc ensemble experiments
│   │   ├── bert_RE_inference.ipynb            # Inference with global threshold tuning
│   │   ├── bert_RE_inference_predicate_specific.ipynb  # Per-predicate threshold tuning
│   │   └── bert_RE_inference_unified.ipynb    # Unified inference pipeline (R1 + R2)
│   │
│   ├── evaluate.py                            # Official evaluation script (2026)
│
├── README.md
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/sofiamaule/SMTE-GutBrainIE.git
cd GutBrainIE

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Evaluation

NER: a prediction is correct only if both label and character offsets match exactly.
M-RE: a prediction is correct only if subject mention, predicate, and object mention all match exactly.

```bash
# NER evaluation
python src/evaluate.py --task ner --predictions <predictions_file> --gold <gold_file>

# RE evaluation
python src/evaluate.py --task re --predictions <predictions_file> --gold <gold_file>
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{maule2026smte,
  author    = {Maule, Sofia and Di Nunzio, Giorgio Maria},
  title     = {SMTE -- GutBrainIE@CLEF 2026: Biomedical Named Entity Recognition
               and Mention-Level Relation Extraction for the Gut-Brain Axis},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026},
  publisher = {CEUR-WS}
}
```

---

## References

- [GutBrainIE @ CLEF 2026](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/)
- [BioASQ CLEF Lab 2026](https://www.bioasq.org/workshop2026)
- Lee et al. (2020). BioBERT. *Bioinformatics* 36, 1234–1240.
- Gu et al. (2021). Domain-specific language model pretraining for biomedical NLP. *ACM THMS* 3, 1–23.
- Baldini Soares et al. (2019). Matching the Blanks. *ACL 2019*, 2895–2905.
- Yasunaga et al. (2022). LinkBERT. *ACL 2022*, 8003–8016.
