#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RAG on multi-sample JSONL produced by multi_sample_infer.py.
Each input line is a sample; output keeps original fields and enriches ingredients with
canonical_name / food_key / food_name.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parents[3]))
RAG_DIR = ROOT / "RAG"
CANONICAL_RUN = RAG_DIR / "canonical" / "rag_run.py"
FOODKEY_RUN = RAG_DIR / "foodkey" / "resolve_foodkey.py"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_rag_map(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in _iter_jsonl(path):
        sid = str(item.get("sample_id", "")).strip()
        try:
            idx = int(item.get("ingredient_index", -1))
        except Exception:
            continue
        if sid and idx >= 0:
            out[(sid, idx)] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG on multi-sample JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL from multi_sample_infer.py")
    parser.add_argument("--output", required=True, help="Output JSONL with RAG fields")
    parser.add_argument("--device", default="cuda:0", help="Device for resolve_foodkey")
    parser.add_argument("--tmp_dir", default=str(CURRENT_DIR / "_rag_tmp"), help="Temp dir")
    parser.add_argument("--kb", default=None, help="Optional KB path for resolve_foodkey")
    parser.add_argument("--emb_model_dir", default=None, help="Override embedding model dir")
    parser.add_argument("--expand_local_model_path", default=None, help="Override local LLM path for expand")
    parser.add_argument("--max_samples", type=int, default=None, help="Process only first N samples")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(_iter_jsonl(in_path)):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        sample_id = sample.get("sample_id") or f"sample_{idx}"
        sample["sample_id"] = sample_id
        ingredients = sample.get("ingredients", []) or []
        for ing_idx, ing in enumerate(ingredients):
            if not isinstance(ing, dict):
                continue
            flat_rows.append({
                "sample_id": sample_id,
                "ingredient_index": ing_idx,
                "name": ing.get("name", ""),
                "note": ing.get("note", ""),
                "ratio": ing.get("ratio", None),
            })
        samples.append(sample)

    if not flat_rows:
        _write_jsonl(out_path, samples)
        return

    rag_in = tmp_dir / "rag_input.jsonl"
    rag_canonical = tmp_dir / "rag_canonical.jsonl"
    rag_foodkey = tmp_dir / "rag_foodkey.jsonl"

    _write_jsonl(rag_in, flat_rows)

    env = dict(os.environ)
    device_norm = (args.device or "").lower()
    if device_norm.startswith("cpu"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device_norm.startswith("cuda") and ":" in device_norm:
        env["CUDA_VISIBLE_DEVICES"] = device_norm.split(":", 1)[1]
    if args.emb_model_dir:
        env["EMB_MODEL_DIR"] = args.emb_model_dir
    if args.expand_local_model_path:
        env["EXPAND_LOCAL_MODEL_PATH"] = args.expand_local_model_path

    subprocess.run(
        [sys.executable, str(CANONICAL_RUN), "--input", str(rag_in), "--output", str(rag_canonical)],
        check=True,
        env=env,
    )

    cmd = [
        sys.executable,
        str(FOODKEY_RUN),
        "--input",
        str(rag_canonical),
        "--output",
        str(rag_foodkey),
        "--device",
        args.device,
    ]
    kb_path = args.kb or str(ROOT / "RAG/ausnut_kb_measures.tagged.jsonl")
    if kb_path:
        cmd += ["--kb", kb_path]
    subprocess.run(cmd, check=True, env=env)

    rag_map = _load_rag_map(rag_foodkey)

    out_rows: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", "")).strip()
        ingredients = sample.get("ingredients", []) or []
        out_ingredients = []
        for ing_idx, ing in enumerate(ingredients):
            if not isinstance(ing, dict):
                continue
            rag = rag_map.get((sample_id, ing_idx), {})
            out_ingredients.append({
                "name": ing.get("name", ""),
                "note": ing.get("note", ""),
                "ratio": ing.get("ratio", None),
                "canonical_name": rag.get("canonical_name", ""),
                "food_key": rag.get("food_key", ""),
                "food_name": rag.get("food_name", ""),
            })
        out_row = dict(sample)
        out_row["ingredients"] = out_ingredients
        out_rows.append(out_row)

    _write_jsonl(out_path, out_rows)


if __name__ == "__main__":
    main()
