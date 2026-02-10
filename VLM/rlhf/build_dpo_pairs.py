#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DPO pairs from:
  - true samples (gold)
  - multi_samples.jsonl (raw generations)
  - multi_samples_rag.jsonl (rag-annotated with food_key)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
        content = match.group(1) if match else text
        return json.loads(content)
    except Exception:
        return {}


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


def _gold_to_text(gold: Dict[str, Any], lowercase_name: bool) -> str:
    title = str(gold.get("title", "")).strip()
    ingredients = gold.get("ingredients", []) or []
    out_ings = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        name, note = _build_name_note(ing, lowercase_name)
        if not name:
            continue
        out_ings.append({"name": name, "note": note})
    obj = {"title": title, "ingredients": out_ings}
    return json.dumps(obj, ensure_ascii=False)


def _candidate_to_text(candidate: Dict[str, Any]) -> str:
    gen = candidate.get("generated_text", "")
    if isinstance(gen, str) and gen.strip():
        obj = _extract_json(gen)
        if isinstance(obj, dict) and "ingredients" in obj:
            return json.dumps(obj, ensure_ascii=False)
    # fallback: build from fields
    title = str(candidate.get("title", "")).strip()
    ingredients = candidate.get("ingredients", []) or []
    out_ings = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        name = str(ing.get("name", "")).strip()
        note = str(ing.get("note", "")).strip()
        if not name:
            continue
        out_ings.append({"name": name, "note": note})
    obj = {"title": title, "ingredients": out_ings}
    return json.dumps(obj, ensure_ascii=False)


def _score_match(gold_keys: List[str], cand_keys: List[str]) -> float:
    g = {k for k in gold_keys if k}
    c = {k for k in cand_keys if k}
    if not g:
        return 0.0
    return len(g.intersection(c)) / float(len(g))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO pairs from gold + candidates")
    parser.add_argument("--true", required=True, help="true samples (.json or .jsonl)")
    parser.add_argument("--candidates", required=True, help="multi_samples.jsonl")
    parser.add_argument("--candidates_rag", required=True, help="multi_samples_rag.jsonl")
    parser.add_argument("--output", required=True, help="output dpo jsonl")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text")
    parser.add_argument("--lowercase_name", action="store_true", help="lowercase ingredient names for chosen")
    parser.add_argument("--max_pairs", type=int, default=None, help="limit number of pairs")
    args = parser.parse_args()

    true_path = Path(args.true).expanduser().resolve()
    cand_path = Path(args.candidates).expanduser().resolve()
    cand_rag_path = Path(args.candidates_rag).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_by_image: Dict[str, Dict[str, Any]] = {}
    for item in _iter_json_or_jsonl(true_path):
        image = item.get("image") or item.get("images", [None])[0]
        if image:
            gold_by_image[str(image)] = item

    cand_by_id: Dict[str, Dict[str, Any]] = {}
    cand_ids_by_image: Dict[str, List[str]] = {}
    for item in _iter_json_or_jsonl(cand_path):
        sid = str(item.get("sample_id", "")).strip()
        if not sid:
            continue
        cand_by_id[sid] = item
        image = str(item.get("image", "")).strip()
        if image:
            cand_ids_by_image.setdefault(image, []).append(sid)

    cand_keys_by_id: Dict[str, List[str]] = {}
    for item in _iter_json_or_jsonl(cand_rag_path):
        sid = str(item.get("sample_id", "")).strip()
        if not sid:
            continue
        keys = []
        for ing in item.get("ingredients", []) or []:
            if isinstance(ing, dict):
                k = str(ing.get("food_key", "")).strip()
                if k:
                    keys.append(k)
        cand_keys_by_id[sid] = keys

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for image, gold in gold_by_image.items():
            gold_keys = []
            for ing in gold.get("ingredients", []) or []:
                if isinstance(ing, dict):
                    k = str(ing.get("food_key", "")).strip()
                    if k:
                        gold_keys.append(k)

            cand_ids = cand_ids_by_image.get(image, [])
            if len(cand_ids) < 2:
                continue

            scored: List[Tuple[str, float]] = []
            for sid in cand_ids:
                keys = cand_keys_by_id.get(sid, [])
                score = _score_match(gold_keys, keys)
                scored.append((sid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            if not scored:
                continue

            # chosen is gold; rejected should be the best (closest) candidate that is still worse than gold.
            # if all candidates perfectly match gold (score == 1), skip.
            rejected_id = None
            rejected_score = None
            for sid, score in scored:
                if score < 1.0:
                    rejected_id = sid
                    rejected_score = score
                    break
            if rejected_id is None or rejected_score is None:
                continue

            chosen_id, chosen_score = scored[0]

            chosen_text = _gold_to_text(gold, args.lowercase_name)
            rejected_text = _candidate_to_text(cand_by_id[rejected_id])

            row = {
                "images": [image],
                "messages": [
                    {"role": "user", "content": f"<image>\n{args.prompt}"},
                    {"role": "assistant", "content": chosen_text},
                ],
                "rejected_response": rejected_text,
                "meta": {
                    "chosen_sample_id": chosen_id,
                    "rejected_sample_id": rejected_id,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                },
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.max_pairs is not None and n >= args.max_pairs:
                break


if __name__ == "__main__":
    main()
