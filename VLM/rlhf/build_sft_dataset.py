#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SFT JSONL from gold/true samples.
Converts canonical_name -> name, and (food_name - canonical_name) -> note.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PROMPT = (
    "Given a food image, output only the recipe title and ingredient list in JSON format.\n"
    "The JSON should look like this:\n"
    "{\"title\": \"<title>\", \"ingredients\": [{\"name\": \"<name>\", \"note\": \"<note>\"}, ...]}\n"
    "Do not output cuisine/meal/dish labels, amounts, or ratios."
)


def _iter_json_or_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(obj, dict):
                yield obj


def _strip_canonical(food_name: str, canonical: str) -> str:
    if not food_name:
        return ""
    if not canonical:
        return food_name.strip()
    pattern = re.compile(r"^\s*" + re.escape(canonical) + r"\s*[,;:-]?\s*", re.I)
    out = pattern.sub("", food_name, count=1).strip()
    if not out:
        return ""
    return out.strip(" ,;:-")


def _build_name_note(ing: Dict[str, Any], lowercase_name: bool) -> Tuple[str, str]:
    canonical = str(ing.get("canonical_name", "")).strip()
    food_name = str(ing.get("food_name", "")).strip()
    name = canonical or str(ing.get("name", "")).strip() or food_name
    if lowercase_name:
        name = name.lower()
    note = _strip_canonical(food_name, canonical)
    if not note and ing.get("note"):
        note = str(ing.get("note", "")).strip()
    return name, note


def _build_assistant_json(title: str, ingredients: List[Dict[str, Any]], lowercase_name: bool) -> Dict[str, Any]:
    out_ings = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        name, note = _build_name_note(ing, lowercase_name)
        if not name:
            continue
        out_ings.append({"name": name, "note": note})
    return {"title": title, "ingredients": out_ings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT JSONL from true samples")
    parser.add_argument("--input", required=True, help="true samples (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="output SFT jsonl")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="user prompt text")
    parser.add_argument("--lowercase_name", action="store_true", help="lowercase ingredient names")
    parser.add_argument("--wrap_code_fence", action="store_true", help="wrap assistant JSON in ```json fences")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in _iter_json_or_jsonl(in_path):
        image = item.get("image") or item.get("images", [None])[0]
        title = str(item.get("title", "")).strip()
        ingredients = item.get("ingredients", []) or []
        assistant_obj = _build_assistant_json(title, ingredients, args.lowercase_name)
        assistant_text = json.dumps(assistant_obj, ensure_ascii=False)
        if args.wrap_code_fence:
            assistant_text = f\"\"\"```json\n{assistant_text}\n```\"\"\"

        row = {
            "images": [image] if image else [],
            "messages": [
                {"role": "user", "content": f"<image>\\n{args.prompt}"},
                {"role": "assistant", "content": assistant_text},
            ],
        }

        if isinstance(item.get("total_weight"), (int, float)):
            row["total_weight"] = float(item["total_weight"])

        rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
