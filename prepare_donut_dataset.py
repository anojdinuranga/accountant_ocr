import json
import os
import base64
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# === Donut dataset output ===
DATASET_DIR = Path("dataset_donut")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TRAIN_IMG_DIR = TRAIN_DIR / "images"
VAL_IMG_DIR = VAL_DIR / "images"
TRAIN_META = TRAIN_DIR / "metadata.jsonl"
VAL_META = VAL_DIR / "metadata.jsonl"

# === Source directory (all your DocAI JSON files) ===
COLLECTED_JSON_DIR = Path("collected_json")
CURRENT_DIR = Path(".")

# Special fields for fund tables
EMP_TABLE_CHILD_FIELDS = [
    "member_no",
    "nic_no",
    "total_earning",
    "total_contribution",
    "total_contribution_cent",
    "employer_contribution",
    "employee_contribution",
    "employee_contribution_cent",
    "employer_contribution_cent",
    "employee_name",
]

@dataclass
class Example:
    image_path: Path
    label: Dict[str, Any]
    source_json: Path


def iter_json_files(root_dirs: List[Path]) -> Iterable[Path]:
    """Scan for JSON files in given directories."""
    for root_dir in root_dirs:
        if not root_dir.exists():
            continue
        for p in root_dir.rglob("*.json"):
            yield p


def safe_get(dct: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalized_or_mention(entity: Dict[str, Any]) -> Optional[str]:
    val = safe_get(entity, ["normalizedValue", "text"]) or safe_get(entity, ["mentionText"])
    if isinstance(val, str):
        return val
    return None


def extract_employees_fund_rows(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        if ent.get("type") != "employees_fund_table":
            continue
        props = ent.get("properties") or []
        row: Dict[str, Any] = {}
        for child in props:
            child_type = child.get("type")
            if child_type in EMP_TABLE_CHILD_FIELDS:
                row[child_type] = normalized_or_mention(child)
        if any(v for v in row.values()):
            rows.append(row)
    return rows


def extract_all_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract all entities and organize them by type."""
    all_entities: Dict[str, Any] = {}
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        entity_type = ent.get("type")
        if not entity_type:
            continue

        if entity_type == "employees_fund_table":
            rows = extract_employees_fund_rows([ent])
            if rows:
                all_entities.setdefault("employees_fund_table", []).extend(rows)
            continue

        value = normalized_or_mention(ent)
        if value is not None:
            if entity_type in all_entities:
                if not isinstance(all_entities[entity_type], list):
                    all_entities[entity_type] = [all_entities[entity_type]]
                all_entities[entity_type].append(value)
            else:
                all_entities[entity_type] = value
    return all_entities


def extract_page_image(data: Dict[str, Any], jf: Path) -> Optional[Path]:
    """Try extracting base64 image from DocAI JSON."""
    pages = data.get("pages", [])
    if not pages:
        return None
    page = pages[0]  # only first page (extend later for multi-page if needed)
    image_info = page.get("image", {})

    if "content" in image_info:  # base64
        try:
            img_bytes = base64.b64decode(image_info["content"])
            out_path = jf.parent / f"{jf.stem}_page0.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            return out_path
        except Exception as e:
            print(f"Error decoding base64 for {jf.name}: {e}")
            return None

    if "uri" in image_info:
        uri = image_info["uri"]
        print(f"Image for {jf.name} stored remotely ({uri}), skipping.")
        return None
    return None


def find_image_for_json(json_file: Path) -> Optional[Path]:
    """Fallback: try to guess local *_blob_*.png images near the JSON file."""
    stem = json_file.stem
    parent = json_file.parent
    candidates = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidates.extend(parent.glob(f"{stem}_blob_*{ext}"))
        candidates.extend(parent.glob(f"{stem}-formated_blob_*{ext}"))
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def build_examples(root_dirs: List[Path]) -> List[Example]:
    examples: List[Example] = []
    json_files = list(iter_json_files(root_dirs))
    print(f"Found {len(json_files)} JSON files to process")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            continue

        entities = data.get("entities") or []
        if not entities:
            continue
        all_entities = extract_all_entities(entities)
        if not all_entities:
            continue

        label = {
            **all_entities,
            "doc_id": jf.name,
            "source_path": str(jf),
        }

        img = extract_page_image(data, jf)
        if img is None:
            img = find_image_for_json(jf)
        if img is None:
            print(f"⚠️ Skipping {jf.name}: no image found")
            continue

        examples.append(Example(image_path=img, label=label, source_json=jf))
    print(f"Built {len(examples)} examples (with images)")
    return examples


def ensure_dirs():
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def train_val_split(items: List[Example], val_ratio: float = 0.1) -> Tuple[List[Example], List[Example]]:
    n = len(items)
    val_count = max(1, int(n * val_ratio)) if n > 1 else 0
    return items[val_count:], items[:val_count]  # train, val


def copy_and_relpath(img: Path, dest_dir: Path) -> str:
    dest = dest_dir / img.name
    if not dest.exists():
        shutil.copy2(img, dest)
    return f"images/{dest.name}"


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    scan_dirs = [CURRENT_DIR]
    if COLLECTED_JSON_DIR.exists():
        scan_dirs.append(COLLECTED_JSON_DIR)

    ensure_dirs()
    examples = build_examples(scan_dirs)
    if not examples:
        print("No examples created!")
        return

    train_items, val_items = train_val_split(examples, val_ratio=0.1)

    train_recs = []
    val_recs = []

    for ex in train_items:
        rel = copy_and_relpath(ex.image_path, TRAIN_IMG_DIR)
        train_recs.append({
            "image": rel,
            "text": json.dumps(ex.label, ensure_ascii=False)
        })
    for ex in val_items:
        rel = copy_and_relpath(ex.image_path, VAL_IMG_DIR)
        val_recs.append({
            "image": rel,
            "text": json.dumps(ex.label, ensure_ascii=False)
        })

    write_jsonl(TRAIN_META, train_recs)
    write_jsonl(VAL_META, val_recs)

    print(f"\n✅ Done! Donut dataset created.")
    print(f"Train examples: {len(train_recs)}, Val examples: {len(val_recs)}")
    print(f"Train metadata: {TRAIN_META}")
    print(f"Val metadata:   {VAL_META}")


if __name__ == "__main__":
    main()
