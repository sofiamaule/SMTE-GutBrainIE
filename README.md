# GutBrainIE — NER & Relation Extraction for the Gut-Brain Axis

This repository contains the implementation of **SMTE**, a system developed for **Task 6 of the BioASQ CLEF Lab 2026 — GutBrainIE**, a Natural Language Processing challenge focused on extracting structured information from biomedical abstracts related to the gut microbiota and its connections with neurological and mental health conditions (Alzheimer's, Parkinson's, Multiple Sclerosis, ALS, and mental health disorders).

The project is part of the EU-supported [HEREDITARY](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) project.

Two subtasks are addressed:

- **Subtask 6.1.1** — Named Entity Recognition (NER)
- **Subtask 6.2.1** — Mention-Level Relation Extraction (M-RE)

---

## Tasks

### Subtask 6.1.1 — Named Entity Recognition (NER)

Given a PubMed abstract discussing the gut-brain interplay, the system must identify and classify specific text spans (entity mentions) into one of **13 predefined biomedical categories**: `anatomical location`, `animal`, `bacteria`, `biomedical technique`, `chemical`, `DDF` (disease, disorder, or finding), `dietary supplement`, `drug`, `food`, `gene`, `human`, `microbiome`, `statistical technique`.

Each prediction is expressed as a tuple:

```
(entityCategory ; entityLocation ; startOffset ; endOffset)
```

where `entityLocation` indicates whether the mention appears in the `title` or `abstract`, and `startOffset`/`endOffset` are character-level offsets.

**Final system:** A hybrid ensemble combining **BioBERT** and **PubMedBERT** fine-tuned on Gold + Silver + Bronze data, achieving **Macro-F1 0.802** and **Micro-F1 0.8324** on the development set.

**Training strategies explored:**

| Config | Description |
|--------|-------------|
| NB1 | Single-stage training on Gold + Silver (BioBERT) |
| NB1b | Single-stage on Gold + Silver + Bronze (BioBERT / PubMedBERT) — **best single model** |
| NB2 | Two-stage curriculum training: Gold first, then Gold+Silver (BioBERT) |
| NB3 | Three-stage cross-year curriculum: 2025 data → 2026 Gold → 2026 Gold+Silver |
| NB4 | Annotator-weighted training on Gold + Silver + Silver 2025 |
| NB4b | Annotator-weighted training including Bronze |

**Inference pipeline:** BIO decoding with confidence scoring (median token probability), label-specific two-pass thresholding, label-aware post-processing heuristics, and deduplication with overlap resolution.

---

### Subtask 6.2.1 — Mention-Level Relation Extraction (M-RE)

Given a PubMed abstract, the system must identify **relations between specific entity mentions** and assign the correct relation predicate. The schema defines **17 semantic predicates** (`influence`, `is linked to`, `change abundance`, `impact`, `administered`, `affect`, `change effect`, `change expression`, `compared to`, `interact`, `is a`, `located in`, `part of`, `produced by`, `strike`, `target`, `used by`) over 55 legal type-pair patterns across 52 unique subject–object type combinations.

Each prediction is expressed as a triple:

```
(subjectMention ; relationPredicate ; objectMention)
```

**Final system:** **PubMedBERT-large** fine-tuned with hard negative sampling, mention-mean pooling, and full-context input (A5-fullctx), achieving **Macro-F1 0.5149** and **Micro-F1 0.5961** on the development set (+0.0208 vs. baseline).

**Training configurations explored:**

| Config | Description |
|--------|-------------|
| A0 | Baseline: marker-only pooling, uniform negative sampling (BiomedBERT-base) |
| A2 | Weighted loss by annotation quality tier |
| A3 | Mention-mean pooling (replaces marker-only) |
| A4 | Label smoothing (ε=0.1) on top of A3 |
| A5 | Hard negative sampling: 70% hard negatives + 30% easy (BiomedBERT-base / PubMedBERT-large) — **best base-model config** |
| A5-fullctx | A5-large with full title–abstract context window (±10,000 chars) — **final submitted system** |
| A5-BioLink | A5 setup with BioLinkBERT-base backbone |
| A6-typed | Typed entity markers [ENTITY_TYPE]...[/ENTITY_TYPE] — 26 new special tokens |
| A7 | Typed markers + hard negatives |

**Inference pipeline:** Logit caching, legal-only decoding with temperature scaling (T=1.25), distance-aware minimum probability floor, per-predicate threshold tuning (margin, min_prob, max_chars, top2_gap), and deduplication.

---

## Dataset

The training data is provided in **JSON format**, organized into four quality tiers:

| Tier | Description |
|------|-------------|
| **Gold** | High-quality, expert-curated annotations (7 experts + 6 external contributors) |
| **Silver** | Mid-quality annotations by ~55 trained linguistics students (supervised by experts) |
| **Silver 2025** | Annotations from the 2025 edition; concept-level annotations are auto-generated |
| **Bronze** | Distantly supervised, fully automatic annotations (GLiNER for NER, ATLOP for RE) |

The test set consists of a held-out subset of the Gold data. The training set for RE contains 4,921 documents (639 Gold, 811 Silver, 499 Silver 2025, 2,972 Bronze), yielding 53,791 positive and 194,662 negative relation examples.

Each JSON entry includes:
- **Metadata**: PMID, title, abstract, author, journal, year, annotator ID
- **Entities**: text span, label, start/end offsets, location, concept URI
- **Relations**: subject/object spans with labels, predicate, offsets, and concept URIs

---

## Results Summary

### NER — Development Set

| Model | Mac-F1 | Mic-F1 |
|-------|--------|--------|
| NB1 (BioBERT, Gold+Silver) | 0.7648 | 0.8070 |
| NB1b (BioBERT, +Bronze) | 0.7749 | 0.8194 |
| NB1b (PubMedBERT, +Bronze) | 0.7973 | 0.8270 |
| **Ensemble (final)** | **0.802** | **0.8324** |

### M-RE — Development Set

| Model | Mac-F1 | Mic-F1 |
|-------|--------|--------|
| A0 (BiomedBERT-base, baseline) | 0.4981 | 0.5753 |
| A5 (BiomedBERT-base, hard neg) | 0.5142 | 0.5899 |
| A5-large (PubMedBERT-large) | 0.5155 | 0.5932 |
| **A5-fullctx (final)** | **0.5149** | **0.5961** |

---

## Repository Structure

```
GutBrainIE/
│
├── data/
│   ├── GutBrainIE_Full_Collection_2025/       # Training data (2025 edition)
│   └── GutBrainIE_Full_Collection_2026/       # Training data (2026 edition)
│
├── src/
│   ├── ner/                                   # NER subtask (6.1.1)
│   │   ├── gliner/                            # GLiNER-based NER experiments
│   │   ├── models/                            # Saved/fine-tuned NER models
│   │   ├── old_notebooks_2025/                # Notebooks trained on 2025 collection
│   │   ├── predictions/                       # NER inference outputs
│   │   │
│   │   ├── bert_NER_NB1_twopass.ipynb         # NB1: Gold+Silver, single-stage
│   │   ├── bert_NER_NB2_twopass_stage_training.ipynb     # NB2: Two-stage curriculum
│   │   ├── bert_NER_NB3_twopass_weight_train2526.ipynb   # NB3: Cross-year curriculum
│   │   ├── bert_NER_NB4_twopass_weighted_annotators.ipynb        # NB4: Annotator weights
│   │   ├── bert_NER_NB4b_twopass_weighted_annotators_bronze.ipynb # NB4b: +Bronze
│   │   ├── bert_NER_ensemble.ipynb            # Final ensemble (NB1b BioBERT + PubMedBERT)
│   │   └── bert_NER_inference_all_models.ipynb
│   │
│   ├── re/                                    # Relation Extraction subtask (6.2.1)
│   │   ├── models/                            # Saved/fine-tuned RE models
│   │   ├── predictions/                       # RE inference outputs
│   │   │
│   │   ├── training/
│   │   │   ├── bert_RE.ipynb                  # Baseline
│   │   │   ├── bert_RE_A0_fixed.ipynb         # A0: marker-only, uniform negatives
│   │   │   ├── bert_RE_A2_weighted_fixed.ipynb  # A2: quality-weighted loss
│   │   │   ├── bert_RE_A3_mentionmean.ipynb   # A3: mention-mean pooling
│   │   │   ├── bert_RE_A4_labelsmooth.ipynb   # A4: label smoothing
│   │   │   ├── bert_RE_A5_hardneg.ipynb       # A5: hard negative sampling
│   │   │   ├── bert_RE_A5_fullctx.ipynb       # A5-fullctx: full context — FINAL SYSTEM
│   │   │   ├── bert_RE_A6_typed.ipynb         # A6: typed entity markers
│   │   │   └── bert_RE_A7_typed_hardneg.ipynb # A7: typed markers + hard negatives
│   │   │
│   │   ├── bert_RE_ensemble.ipynb             # Post-hoc ensemble experiments
│   │   ├── bert_RE_inference.ipynb            # Inference with global threshold tuning
│   │   ├── bert_RE_inference_predicate_specific.ipynb  # Per-predicate threshold tuning
│   │   └── bert_RE_inference_unified.ipynb    # Unified inference pipeline
│   │
│   ├── evaluate.py                            # Evaluation script (2026)
│   └── evaluate2025.py                        # Evaluation script (2025)
│
├── venv/                                      # Python virtual environment
├── .gitignore
├── GutBrainIE_SofiaMaule.pdf                 # Project report (CLEF 2026 working notes)
├── RE_Tabella_e_TODO_v3.pdf                  # RE experiments table and TODO notes
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/sofiamaule/GutBrainIE.git
cd GutBrainIE

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Evaluation

The evaluation scripts follow the official GutBrainIE scoring protocol. A NER prediction is correct only if both label and character offsets match exactly; an M-RE prediction is correct only if subject mention, predicate, and object mention all match exactly.

```bash
# Evaluate NER predictions (2026)
python src/evaluate.py --task ner --predictions <predictions_file> --gold <gold_file>

# Evaluate RE predictions (2026)
python src/evaluate.py --task re --predictions <predictions_file> --gold <gold_file>

# Evaluate on 2025 data
python src/evaluate2025.py --task ner --predictions <predictions_file> --gold <gold_file>
```

---

## References

- [GutBrainIE @ CLEF 2026 — Task Page](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/)
- [BioASQ CLEF Lab 2026](https://www.bioasq.org/)
- Lee et al. (2020). BioBERT: a pre-trained biomedical language representation model. *Bioinformatics* 36, 1234–1240.
- Gu et al. (2021). Domain-specific language model pretraining for biomedical NLP. *ACM THMS* 3, 1–23.
