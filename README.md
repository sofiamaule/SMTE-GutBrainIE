# GutBrainIE — NER & Relation Extraction for the Gut-Brain Axis

This repository contains the implementation developed for **Task 6 of the BioASQ CLEF Lab 2026 — GutBrainIE**, a Natural Language Processing challenge focused on extracting structured information from biomedical abstracts related to the gut microbiota and its connections with neurological and mental health conditions (Alzheimer's, Parkinson's, Multiple Sclerosis, ALS, and mental health disorders).

The project is part of the EU-supported [HEREDITARY](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/) project.

Two subtasks are addressed:

- **Subtask 6.1.1** — Named Entity Recognition (NER)
- **Subtask 6.2.1** — Mention-Level Relation Extraction (M-RE)

---

## Tasks

### Subtask 6.1.1 — Named Entity Recognition (NER)

Given a PubMed abstract discussing the gut-brain interplay, the system must identify and classify specific text spans (entity mentions) into one of **13 predefined biomedical categories** (e.g., `bacteria`, `chemical`, `microbiota`, `disease`, ...).

Each prediction is expressed as a tuple:

```
(entityCategory ; entityLocation ; startOffset ; endOffset)
```

where `entityLocation` indicates whether the mention appears in the `title` or `abstract`, and `startOffset`/`endOffset` are character-level offsets.

**Approach:** Fine-tuned BERT-based token classification models, with experiments including:
- Two-pass inference strategies
- Weighted training over annotation quality tiers (Gold / Silver / Bronze)
- Annotator-aware weighting
- Ensemble methods

---

### Subtask 6.2.1 — Mention-Level Relation Extraction (M-RE)

Given a PubMed abstract, the system must identify **relations between specific entity mentions** and assign the correct relation predicate connecting them.

Each prediction is expressed as a triple:

```
(subjectMention ; relationPredicate ; objectMention)
```

**Approach:** Fine-tuned BERT-based relation classification models, with experiments including:
- Weighted training strategies
- Hard negative mining
- Typed entity markers
- Label smoothing
- Margin-based loss
- LLM fallback (via GROQ API)
- Unified micro-F1 inference

---

## Dataset

The training data is provided in **JSON format**, organized into four quality tiers:

| Tier | Description |
|------|-------------|
| **Gold** | High-quality, expert-curated annotations (7 experts + 6 external contributors) |
| **Silver** | Mid-quality annotations by ~55 trained linguistics students (supervised by experts) |
| **Silver 2025** | Same as Silver, for the 2025 edition; concept-level annotations are auto-generated |
| **Bronze** | Distantly supervised, fully automatic annotations (GLiNER for NER, ATLOP for RE) |

Each JSON entry corresponds to a PubMed article and includes:
- **Metadata**: PMID, title, abstract, author, journal, year, annotator ID
- **Entities**: text span, label, start/end offsets, location, concept URI
- **Relations**: subject/object spans with labels, predicate, offsets, and concept URIs

---

## Repository Structure

```
GutBrainIE/
│
├── data/
│   ├── GutBrainIE_Full_Collection_2025/   # Training data (2025 edition)
│   └── GutBrainIE_Full_Collection_2026/   # Training data (2026 edition)
│
├── src/
│   ├── ner/                               # NER subtask (6.1.1)
│   ├── gliner/                            # GLiNER-based NER experiments
│   │   ├── models/                        # Saved/fine-tuned NER models
│   │   ├── old_notebooks_2025/            # notebooks trained on 2025 colection
│   │   ├── predictions/                   # NER inference outputs
│   │   │
│   │   ├── bert_NER_ensemble.ipynb
│   │   ├── bert_NER_inference_all_models.ipynb
│   │   ├── bert_NER_NB1_twopass.ipynb
│   │   ├── bert_NER_NB2_twopass_stage_training.ipynb
│   │   ├── bert_NER_NB3_twopass_weight_train2526.ipynb
│   │   ├── bert_NER_NB4_twopass_weighted_annotators.ipynb
│   │   └── bert_NER_NB4b_twopass_weighted_annotators_bronze.ipynb
│   │
│   ├── re/                                # Relation Extraction subtask (6.2.1)
│   │   ├── models/                        # Saved/fine-tuned RE models
│   │   ├── predictions/                   # RE inference outputs
│   │   │
│   │   ├── training/                      # Training notebooks
│   │   │   ├── bert_RE.ipynb              # Baseline
│   │   │   ├── bert_RE_A0_fixed.ipynb
│   │   │   ├── bert_RE_A2_weighted_fixed.ipynb
│   │   │   ├── bert_RE_A3_mentionmean.ipynb
│   │   │   ├── bert_RE_A4_labelsmooth.ipynb
│   │   │   ├── bert_RE_A5_hardneg.ipynb
│   │   │   ├── bert_RE_A6_typed.ipynb
│   │   │   └── bert_RE_A7_typed_hardneg.ipynb
│   │   │
│   │   # --- Inference notebooks ---
│   │   ├── bert_RE_inference.ipynb
│   │   ├── bert_RE_inference_predicate_specific.ipynb
│   │   ├── bert_RE_inference_unified.ipynb
│   │   ├── bert_RE_inference_unified_micrf1.ipynb
│   │   │
│   │   # --- Additional strategies ---
│   │   ├── bert_RE_llm_fallback.ipynb     # GROQ LLM fallback
│   │   ├── bert_RE_margin.ipynb           # Margin-based loss
│   │
│   ├── evaluate.py                        # Evaluation script (2026)
│   └── evaluate2025.py                    # Evaluation script (2025)
│
├── venv/                                  # Python virtual environment
├── .gitignore
├── GutBrainIE_SofiaMaule.pdf             # Project report
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# Clone the repository
git clone <https://github.com/sofiamaule/GutBrainIE.git>
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

The evaluation scripts follow the official GutBrainIE scoring protocol.

```bash
# Evaluate NER predictions (2026)
python src/evaluate.py --task ner --predictions <predictions_file> --gold <gold_file>

# Evaluate RE predictions (2026)
python src/evaluate.py --task re --predictions <predictions_file> --gold <gold_file>
```

---

## References

- [GutBrainIE @ CLEF 2026 — Task Page](https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/)
- [BioASQ CLEF Lab 2026](https://www.bioasq.org/)
