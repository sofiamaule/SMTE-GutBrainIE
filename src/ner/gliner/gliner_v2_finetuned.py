import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


# ============================================================
# 0) CONFIG (EDIT ME)
# ============================================================
DATA_DIR = Path(r"C:/Users/super/Documents/UniPd/ATA/GutBrainIE/data/GutBrainIE_Full_Collection_2025/Annotations")

# --- Adjust these to your actual files ---
TRAIN_JSONS = [
    DATA_DIR / "Train/gold_quality/json_format/train_gold.json",
    DATA_DIR / "Train/platinum_quality/json_format/train_platinum.json",
    DATA_DIR / "Train/silver_quality/json_format/train_silver.json",
]
DEV_JSON = DATA_DIR / "Dev/json_format/dev.json"

# Base model
BASE_MODEL = "fastino/gliner2-base-v1"

# Output dirs
OUT_MODEL_DIR = Path(r"/ner/models/old/gliner_v2_finetuned")
OUT_PRED_DIR = Path(r"/ner/predictions")

# Inference threshold for dev prediction (tune later!)
THRESHOLD = 0.33
INCLUDE_CONFIDENCE = True

SEED = 42
# -----------------------------
# RUN MODES
# -----------------------------
DO_TRAIN = False          # ✅ metti False per NON rifare training
DO_PREDICT_DEV = True     # lascia True per fare predizioni su dev
ADAPTER_DIR = Path(r"/ner/models/old/gliner_v2_finetuned\best")


# ============================================================
# 1) LABELS
# ============================================================
ENTITY_LABELS: List[str] = [
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "disease, disorder or finding",  # will normalize to DDF
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
]
LABEL_MAPPING = {"disease, disorder or finding": "DDF"}

def normalize_label(label: str) -> str:
    return LABEL_MAPPING.get(label, label)

LEGAL_ENTITY_LABELS = {
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "DDF",
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
}

# (Optional) descrizioni: spesso aiutano i modelli “instruction-like”
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "anatomical location": "Anatomical body part, tissue, organ, or region",
    "animal": "Animal species or animal model names",
    "bacteria": "Bacterial taxa, strains, species, genera",
    "biomedical technique": "Lab/clinical/experimental technique or method",
    "chemical": "Chemical substance, metabolite, compound",
    "DDF": "Disease, disorder, symptom, or clinical finding",
    "dietary supplement": "Supplement, nutraceutical, probiotic product as supplement",
    "drug": "Drug, medication, pharmaceutical substance",
    "food": "Food item, nutrient source, diet components",
    "gene": "Gene symbols / gene products when annotated as gene",
    "human": "Human subjects, patients, cohort descriptors",
    "microbiome": "Microbiota / microbiome mentions (community-level)",
    "statistical technique": "Statistical test/model/analysis method",
}


# ============================================================
# 2) IO HELPERS
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_ner_data(file_paths: List[Path]) -> Dict[str, Any]:
    all_data: Dict[str, Any] = {}
    for fp in file_paths:
        fp = Path(fp)
        if not fp.exists():
            print(f"⚠️ Missing: {fp}")
            continue
        data = load_json(fp)
        all_data.update(data)
        print(f"Loaded {len(data)} docs from {fp.name}")
    if not all_data:
        raise FileNotFoundError("No training data loaded. Check TRAIN_JSONS paths.")
    return all_data

def safe_text_span(text: str, start: int, end_inclusive: int, fallback: str) -> str:
    if text and 0 <= start <= end_inclusive < len(text):
        return text[start:end_inclusive + 1]
    return (fallback or "").strip()


# ============================================================
# 3) BUILD GLINER2 TRAINING EXAMPLES
#    GLiNER2 expects: InputExample(text=..., entities={label:[mentions...]})
# ============================================================
def build_examples(dataset: Dict[str, Any], keep_empty: bool = True) -> List[InputExample]:
    examples: List[InputExample] = []

    for pmid, article in dataset.items():
        title = (article.get("metadata", {}) or {}).get("title", "") or ""
        abstract = (article.get("metadata", {}) or {}).get("abstract", "") or ""
        entities = article.get("entities", []) or []

        # group mentions by location then by label
        by_loc: Dict[str, Dict[str, List[str]]] = {"title": {}, "abstract": {}}

        for e in entities:
            loc = (e.get("location") or "").strip().lower()
            if loc not in ("title", "abstract"):
                continue

            lab_raw = (e.get("label") or "").strip()
            lab = normalize_label(lab_raw)
            if lab not in LEGAL_ENTITY_LABELS:
                continue

            start = int(e.get("start_idx"))
            end = int(e.get("end_idx"))  # inclusive in GutBrainIE
            fallback = (e.get("text_span") or "").strip()

            src_text = title if loc == "title" else abstract
            mention = safe_text_span(src_text, start, end, fallback)
            mention = mention.strip()
            if not mention:
                continue

            by_loc[loc].setdefault(lab, []).append(mention)

        # build separate examples for title/abstract
        for loc in ("title", "abstract"):
            text = title if loc == "title" else abstract
            if not text.strip():
                continue

            ent_map = {
                lab: sorted(set(mentions))
                for lab, mentions in by_loc[loc].items()
                if mentions
            }

            if (not ent_map) :
                continue

            examples.append(
                InputExample(
                    text=text,
                    entities=ent_map,
                )
            )

    return examples


# ============================================================
# 4) YOUR POST-PROCESSING (copied from your baseline)
# ============================================================
def dedupe_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in ents:
        k = (e["start_idx"], e["end_idx"], e["location"], e["label"])
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out

def prune_overlaps_keep_longest(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_loc: Dict[str, List[Dict[str, Any]]] = {}
    for e in ents:
        by_loc.setdefault(e["location"], []).append(e)

    kept_all: List[Dict[str, Any]] = []
    for loc, group in by_loc.items():
        group = sorted(
            group,
            key=lambda x: (x["start_idx"], -(x["end_idx"] - x["start_idx"]), -x.get("score", 1.0)),
        )

        clusters = []
        cur = []
        cur_end = None

        for e in group:
            s, t = e["start_idx"], e["end_idx"]
            if not cur:
                cur = [e]
                cur_end = t
            else:
                if s <= cur_end:  # overlap inclusive
                    cur.append(e)
                    cur_end = max(cur_end, t)
                else:
                    clusters.append(cur)
                    cur = [e]
                    cur_end = t
        if cur:
            clusters.append(cur)

        for cl in clusters:
            best = max(cl, key=lambda x: ((x["end_idx"] - x["start_idx"]), x.get("score", 1.0)))
            kept_all.append(best)

    return kept_all

def merge_adjacent_same_label(ents: List[Dict[str, Any]], title: str, abstract: str) -> List[Dict[str, Any]]:
    by_loc: Dict[str, List[Dict[str, Any]]] = {}
    for e in ents:
        by_loc.setdefault(e["location"], []).append(e)

    merged_all: List[Dict[str, Any]] = []
    for loc, group in by_loc.items():
        text = title if loc == "title" else abstract
        group = sorted(group, key=lambda x: (x["label"], x["start_idx"], x["end_idx"]))

        i = 0
        while i < len(group):
            cur = dict(group[i])
            i += 1
            while i < len(group) and group[i]["label"] == cur["label"]:
                nxt = group[i]
                if nxt["start_idx"] == cur["end_idx"] + 1:
                    cur["end_idx"] = nxt["end_idx"]
                    i += 1
                    continue
                if (
                    nxt["start_idx"] == cur["end_idx"] + 2
                    and text
                    and text[cur["end_idx"] + 1 : cur["end_idx"] + 2] == " "
                ):
                    cur["end_idx"] = nxt["end_idx"]
                    i += 1
                    continue
                break

            if text and 0 <= cur["start_idx"] <= cur["end_idx"] < len(text):
                cur["text_span"] = text[cur["start_idx"] : cur["end_idx"] + 1]
            merged_all.append(cur)

    return merged_all

def postprocess_entities(ents: List[Dict[str, Any]], title: str, abstract: str) -> List[Dict[str, Any]]:
    ents = dedupe_entities(ents)
    ents = prune_overlaps_keep_longest(ents)
    ents = merge_adjacent_same_label(ents, title=title, abstract=abstract)
    ents = dedupe_entities(ents)
    return ents


# ============================================================
# 5) GLiNER2 INFERENCE (same idea as your baseline)
# ============================================================
def gliner2_extract_spans(
    extractor: GLiNER2,
    text: str,
    labels: List[str],
    threshold: float,
    location: str,
    include_confidence: bool = True,
) -> List[Dict[str, Any]]:
    if not text:
        return []

    result = extractor.extract_entities(
        text,
        labels,
        threshold=threshold,
        include_spans=True,
        include_confidence=include_confidence,
    )

    formatted: List[Dict[str, Any]] = []
    for raw_label, items in result.get("entities", {}).items():
        norm_label = normalize_label(raw_label)
        if norm_label not in LEGAL_ENTITY_LABELS:
            continue

        for it in items:
            start = int(it["start"])
            end_exclusive = int(it["end"])
            end_inclusive = end_exclusive - 1

            formatted.append(
                {
                    "start_idx": start,
                    "end_idx": end_inclusive,
                    "location": location,
                    "text_span": it["text"],
                    "label": norm_label,
                    "score": float(it.get("confidence", 1.0)),
                }
            )

    return formatted


def predict_dataset_gliner2(
    extractor: GLiNER2,
    dataset: Dict[str, Any],
    labels: List[str],
    threshold: float,
) -> Dict[str, Any]:
    preds: Dict[str, Any] = {}

    for pmid, article in tqdm(dataset.items(), desc="GLiNER2 inference"):
        title = article["metadata"].get("title", "") or ""
        abstract = article["metadata"].get("abstract", "") or ""

        ents = []
        ents += gliner2_extract_spans(extractor, title, labels, threshold, "title", INCLUDE_CONFIDENCE)
        ents += gliner2_extract_spans(extractor, abstract, labels, threshold, "abstract", INCLUDE_CONFIDENCE)

        ents = postprocess_entities(ents, title=title, abstract=abstract)

        preds[pmid] = {
            "entities": [
                {k: e[k] for k in ["start_idx", "end_idx", "location", "text_span", "label"]}
                for e in ents
            ]
        }

    return preds


# ============================================================
# 6) EXACT-MATCH EVAL (yours)
# ============================================================
from collections import Counter

def evaluate_ner_exact(predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    gt_by_pmid = {}
    gt_label_counts = Counter()

    for pmid, article in ground_truth.items():
        gt_by_pmid[pmid] = set()
        for e in article.get("entities", []):
            tup = (int(e["start_idx"]), int(e["end_idx"]), str(e["location"]), str(e["text_span"]), str(e["label"]))
            gt_by_pmid[pmid].add(tup)
            gt_label_counts[e["label"]] += 1

    pred_label_counts = Counter()
    tp_label_counts = Counter()

    for pmid, doc in predictions.items():
        for e in doc.get("entities", []):
            label = e["label"]
            if label not in LEGAL_ENTITY_LABELS:
                continue
            pred_label_counts[label] += 1
            tup = (int(e["start_idx"]), int(e["end_idx"]), str(e["location"]), str(e["text_span"]), str(e["label"]))
            if tup in gt_by_pmid.get(pmid, set()):
                tp_label_counts[label] += 1

    tp = sum(tp_label_counts.values())
    pred_total = sum(pred_label_counts.values())
    gt_total = sum(gt_label_counts.values())

    micro_p = tp / (pred_total + 1e-10)
    micro_r = tp / (gt_total + 1e-10)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-10)

    labels = list(gt_label_counts.keys())
    macro_p = macro_r = macro_f1 = 0.0
    for lab in labels:
        p = tp_label_counts[lab] / (pred_label_counts[lab] + 1e-10)
        r = tp_label_counts[lab] / (gt_label_counts[lab] + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n = max(1, len(labels))
    macro_p /= n
    macro_r /= n
    macro_f1 /= n

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "tp": float(tp),
        "pred_total": float(pred_total),
        "gt_total": float(gt_total),
    }


# ============================================================
# 7) MAIN: TRAIN -> SAVE -> PREDICT DEV -> EVAL
# ============================================================
def main() -> None:
    set_seed(SEED)

    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_ner_data(TRAIN_JSONS)
    dev_data = load_ner_data([DEV_JSON])

    # Build GLiNER2 examples
    train_examples = build_examples(train_data)
    dev_examples = build_examples(dev_data)

    print(f"Train examples: {len(train_examples)}")
    print(f"Dev examples:   {len(dev_examples)}")

    # Validate datasets
    TrainingDataset(train_examples).validate()
    TrainingDataset(dev_examples).validate()

    # Load base model
    if DO_TRAIN:
        # Load base model
        model = GLiNER2.from_pretrained(BASE_MODEL)

        # Training config (start conservative, then tune)
        cfg = TrainingConfig(
            output_dir=str(OUT_MODEL_DIR),
            experiment_name="gutbrainie_gliner2",
            num_epochs=8,
            batch_size=8,
            encoder_lr=1e-5,
            task_lr=5e-4,
            warmup_ratio=0.1,
            scheduler_type="cosine",
            fp16=True,
            eval_strategy="epoch",
            save_best=True,
            early_stopping=True,
            early_stopping_patience=2,

            # LoRA: faster & safer to iterate
            use_lora=True,
            lora_r=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
            save_adapter_only=True,
        )

        trainer = GLiNER2Trainer(model, cfg)

        print("## TRAINING ##")
        try:
            trainer.train(train_examples, dev_examples)
        except TypeError:
            print("⚠️ This GLiNER2 version doesn't support dev in train(); training without dev.")
            trainer.train(train_examples)
    else:
        print("## SKIPPING TRAINING (DO_TRAIN=False) ##")

    print(f"## LOADING BASE MODEL: {BASE_MODEL} ##")
    ft_model = GLiNER2.from_pretrained(BASE_MODEL)

    print(f"## LOADING LoRA ADAPTER FROM: {ADAPTER_DIR} ##")
    if hasattr(ft_model, "load_adapter"):
        ft_model.load_adapter(str(ADAPTER_DIR))
    elif hasattr(ft_model, "load_lora_adapter"):
        ft_model.load_lora_adapter(str(ADAPTER_DIR))
    else:
        raise RuntimeError("No adapter loading method found (load_adapter/load_lora_adapter).")

    if DO_PREDICT_DEV:
        print("## PREDICT DEV ##")
        preds_dev = predict_dataset_gliner2(ft_model, dev_data, ENTITY_LABELS, THRESHOLD)

        out_path = OUT_PRED_DIR / "gliner_v2_finetuned.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(preds_dev, f, ensure_ascii=False, indent=2)

        print("Saved predictions:", out_path)

        metrics = evaluate_ner_exact(preds_dev, dev_data)
        print("## DEV EXACT MATCH METRICS ##")
        print(json.dumps(metrics, indent=2))
    else:
        print("## SKIPPING DEV PREDICTION (DO_PREDICT_DEV=False) ##")


if __name__ == "__main__":
    main()
