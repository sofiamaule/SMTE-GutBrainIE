import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES_BY_TASK = {
    "T611": [
        PROJECT_ROOT / "src/ner/predictions/test_set/SMTE_T611_R1_NERensemble/SMTE_T611_R1_NERensemble.json",
        PROJECT_ROOT / "src/ner/predictions/test_set/SMTE_T611_R2_NERpubmedbert/SMTE_T611_R2_NERpubmedbert.json",
        PROJECT_ROOT / "src/ner/predictions/test_set/SMTE_T611_R3_NERbiobert/SMTE_T611_R3_NERbiobert.json",
    ],
    "T621": [
        PROJECT_ROOT / "src/re/predictions/test_set/SMTE_T621_R1_REfullctx/SMTE_T621_R1_REfullctx.json",
        PROJECT_ROOT / "src/re/predictions/test_set/SMTE_T621_R2_RElargewindow/SMTE_T621_R2_RElargewindow.json",
    ],
}


NER_REQUIRED_FIELDS = {
    "start_idx",
    "end_idx",
    "location",
    "text_span",
    "label",
}

NER_ALLOWED_FIELDS = NER_REQUIRED_FIELDS

RE_REQUIRED_FIELDS = {
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
}

RE_ALLOWED_FIELDS = RE_REQUIRED_FIELDS


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def load_json(path: Path, errors: list[str]) -> Any | None:
    if not path.exists():
        fail(errors, f"[MISSING FILE] {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        fail(errors, f"[INVALID JSON] {path}: {e}")
        return None


def validate_root_object(path: Path, data: Any, errors: list[str]) -> bool:
    if not isinstance(data, dict):
        fail(errors, f"[ROOT ERROR] {path}: root JSON must be an object/dict.")
        return False

    for pmid, value in data.items():
        if not isinstance(pmid, str):
            fail(errors, f"[PMID ERROR] {path}: PMID key must be string, got {type(pmid).__name__}.")
        if not isinstance(value, dict):
            fail(errors, f"[ENTRY ERROR] {path} / {pmid}: value must be an object/dict.")
            return False

    return True


def validate_t611_ner(path: Path, data: Any, errors: list[str]) -> None:
    if not validate_root_object(path, data, errors):
        return

    for pmid, entry in data.items():
        keys = set(entry.keys())

        if keys != {"entities"}:
            fail(
                errors,
                f"[T611 FIELD ERROR] {path} / {pmid}: expected only key 'entities', found {sorted(keys)}."
            )
            continue

        entities = entry["entities"]
        if not isinstance(entities, list):
            fail(errors, f"[T611 TYPE ERROR] {path} / {pmid}: 'entities' must be a list.")
            continue

        for i, ent in enumerate(entities):
            if not isinstance(ent, dict):
                fail(errors, f"[T611 ENTITY ERROR] {path} / {pmid} / entities[{i}]: must be an object.")
                continue

            ent_keys = set(ent.keys())

            missing = NER_REQUIRED_FIELDS - ent_keys
            extra = ent_keys - NER_ALLOWED_FIELDS

            if missing:
                fail(errors, f"[T611 MISSING FIELD] {path} / {pmid} / entities[{i}]: missing {sorted(missing)}.")

            if extra:
                fail(errors, f"[T611 EXTRA FIELD] {path} / {pmid} / entities[{i}]: extra fields not allowed {sorted(extra)}.")

            if "uri" in ent:
                fail(errors, f"[T611 URI ERROR] {path} / {pmid} / entities[{i}]: T611 must NOT contain 'uri'.")

            if "start_idx" in ent and not isinstance(ent["start_idx"], int):
                fail(errors, f"[T611 TYPE ERROR] {path} / {pmid} / entities[{i}]: start_idx must be int.")

            if "end_idx" in ent and not isinstance(ent["end_idx"], int):
                fail(errors, f"[T611 TYPE ERROR] {path} / {pmid} / entities[{i}]: end_idx must be int.")

            for field in ["location", "text_span", "label"]:
                if field in ent and not isinstance(ent[field], str):
                    fail(errors, f"[T611 TYPE ERROR] {path} / {pmid} / entities[{i}]: {field} must be string.")


def validate_t621_re(path: Path, data: Any, errors: list[str]) -> None:
    if not validate_root_object(path, data, errors):
        return

    for pmid, entry in data.items():
        keys = set(entry.keys())

        if keys != {"mention_level_relations"}:
            fail(
                errors,
                f"[T621 FIELD ERROR] {path} / {pmid}: expected only key 'mention_level_relations', found {sorted(keys)}."
            )
            continue

        relations = entry["mention_level_relations"]
        if not isinstance(relations, list):
            fail(errors, f"[T621 TYPE ERROR] {path} / {pmid}: 'mention_level_relations' must be a list.")
            continue

        for i, rel in enumerate(relations):
            if not isinstance(rel, dict):
                fail(errors, f"[T621 RELATION ERROR] {path} / {pmid} / mention_level_relations[{i}]: must be an object.")
                continue

            rel_keys = set(rel.keys())

            missing = RE_REQUIRED_FIELDS - rel_keys
            extra = rel_keys - RE_ALLOWED_FIELDS

            if missing:
                fail(errors, f"[T621 MISSING FIELD] {path} / {pmid} / mention_level_relations[{i}]: missing {sorted(missing)}.")

            if extra:
                fail(errors, f"[T621 EXTRA FIELD] {path} / {pmid} / mention_level_relations[{i}]: extra fields not allowed {sorted(extra)}.")

            for field in RE_REQUIRED_FIELDS:
                if field in rel and not isinstance(rel[field], str):
                    fail(errors, f"[T621 TYPE ERROR] {path} / {pmid} / mention_level_relations[{i}]: {field} must be string.")


def main() -> None:
    errors: list[str] = []

    for task_id, paths in FILES_BY_TASK.items():
        print(f"\nValidating {task_id}...")

        for path in paths:
            print(f"  Checking: {path.relative_to(PROJECT_ROOT)}")
            data = load_json(path, errors)

            if data is None:
                continue

            if task_id == "T611":
                validate_t611_ner(path, data, errors)
            elif task_id == "T621":
                validate_t621_re(path, data, errors)
            else:
                fail(errors, f"[UNKNOWN TASK] {task_id}")

    if errors:
        print("\n❌ Validation failed:\n")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("\n✅ All JSON files are valid for T611 and T621.")


if __name__ == "__main__":
    main()