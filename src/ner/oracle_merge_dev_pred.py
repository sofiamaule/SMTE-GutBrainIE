"""
Build dev_pred.json for the ORACLE experiment (Exp. 8).

Takes the gold dev file (which has metadata.title/abstract) and replaces the
`entities` field of every document with the NER-ensemble PREDICTIONS on dev.

Prerequisites (run these first, they already point at dev.json):
  1. bert_NER_inference_all_models.ipynb  -> writes
       src/ner/predictions/pred_NB1b_gold_silver_bronze.json          (BioBERT)
       src/ner/predictions/pred_NB1b_pubmed_gold_silver_bronze.json   (PubMedBERT)
  2. bert_NER_ensemble.ipynb              -> writes
       src/ner/predictions/pred_ensemble_best.json                    (= R1 on dev)

Then run this script. Point PRED_PATH at the ensemble strategy file that matches
the R1 run you actually SUBMITTED (pred_ensemble_best.json, or e.g.
pred_ensemble_hybrid_all.json / pred_ensemble_hybrid_recall.json).
"""
import json, copy
from pathlib import Path

# --- edit these three paths ---
DEV_PATH  = Path("../../data/GutBrainIE_Full_Collection_2026/Annotations/Dev/json_format/dev.json")
PRED_PATH = Path("../../src/ner/predictions/pred_ensemble_best.json")   # the R1 ensemble on dev
OUT_PATH  = Path("../../data/GutBrainIE_Full_Collection_2026/Annotations/Dev/json_format/dev_pred.json")

dev  = json.load(open(DEV_PATH,  encoding="utf-8"))   # {pmid: {"metadata": {...}, "entities": [...]}}
pred = json.load(open(PRED_PATH, encoding="utf-8"))   # {pmid: {"entities": [...]}}

dev_pred = copy.deepcopy(dev)
missing = 0
for pmid, art in dev_pred.items():
    if pmid in pred:
        art["entities"] = pred[pmid].get("entities", [])
    else:
        art["entities"] = []          # NER produced nothing for this PMID
        missing += 1

json.dump(dev_pred, open(OUT_PATH, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

n_gold = sum(len(v.get("entities", [])) for v in dev.values())
n_pred = sum(len(v.get("entities", [])) for v in dev_pred.values())
print(f"Docs: {len(dev_pred)} | PMIDs with no NER pred: {missing}")
print(f"Entities  gold={n_gold}  predicted={n_pred}")
print(f"Wrote {OUT_PATH}")
print("Next: in bert_RE_inference_unified.ipynb set DEV_PATH = dev_pred.json,")
print("      DELETE MODEL_DIR/dev_logits_cache.pt, re-run to cell 23 (Micro-F1).")
