#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RAG (canonical + foodkey) on a single VLM JSON output.
"""
import argparse
import json
import subprocess
import sys
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parents[3]))  # /scratch/.../cs16
RAG_DIR = ROOT / "RAG"
CANONICAL_RUN = RAG_DIR / "canonical" / "rag_run.py"
FOODKEY_RUN = RAG_DIR / "foodkey" / "resolve_foodkey.py"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_rag_on_vlm_obj(
    obj: Dict[str, Any],
    *,
    device: str = "cuda:0",
    tmp_dir: Path | None = None,
    kb: str | None = None,
    emb_model_dir: str | None = None,
    expand_local_model_path: str | None = None,
) -> Dict[str, Any]:
    title = obj.get("title", "")
    ingredients = obj.get("ingredients", []) or []

    # Build JSONL for rag_run: one line per ingredient
    rows = []
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        rows.append({
            "name": ing.get("name", ""),
            "note": ing.get("note", ""),
        })

    created_tmp = False
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="rag_tmp_"))
        created_tmp = True
    else:
        tmp_dir = Path(tmp_dir).expanduser().resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)

    rag_in = tmp_dir / "rag_input.jsonl"
    rag_canonical = tmp_dir / "rag_canonical.jsonl"
    rag_foodkey = tmp_dir / "rag_foodkey.jsonl"

    _write_jsonl(rows, rag_in)

    # Run canonical stage
    env = dict(os.environ)
    device_norm = (device or "").lower()
    if device_norm.startswith("cpu"):
        # Force CPU for canonical stage by hiding CUDA
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device_norm.startswith("cuda") and ":" in device_norm:
        # Pin to a single GPU for canonical stage
        env["CUDA_VISIBLE_DEVICES"] = device_norm.split(":", 1)[1]
    if emb_model_dir:
        env["EMB_MODEL_DIR"] = emb_model_dir
    if expand_local_model_path:
        env["EXPAND_LOCAL_MODEL_PATH"] = expand_local_model_path

    subprocess.run(
        [sys.executable, str(CANONICAL_RUN), "--input", str(rag_in), "--output", str(rag_canonical)],
        check=True,
        env=env,
    )

    # Run foodkey stage
    cmd = [
        sys.executable,
        str(FOODKEY_RUN),
        "--input",
        str(rag_canonical),
        "--output",
        str(rag_foodkey),
        "--device",
        device,
    ]
    kb_path = kb or str(ROOT / "RAG/ausnut_kb_measures.tagged.jsonl")
    if kb_path:
        cmd += ["--kb", kb_path]
    subprocess.run(cmd, check=True, env=env)

    # Merge results back to final JSON
    enriched = []
    with rag_foodkey.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enriched.append(json.loads(line))

    out_ingredients = []
    for ing, rag in zip(ingredients, enriched):
        if not isinstance(ing, dict):
            continue
        out_ingredients.append({
            "name": ing.get("name", ""),
            "note": ing.get("note", ""),
            "ratio": ing.get("ratio", None),
            "canonical_name": rag.get("canonical_name", ""),
            "food_key": rag.get("food_key", ""),
            "food_name": rag.get("food_name", ""),
        })

    out_obj = {
        "title": title,
        "ingredients": out_ingredients,
    }

    if created_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_obj


def run_rag_on_vlm_file(
    input_path: Path,
    output_path: Path,
    *,
    device: str = "cuda:0",
    tmp_dir: Path | None = None,
    kb: str | None = None,
    emb_model_dir: str | None = None,
    expand_local_model_path: str | None = None,
) -> None:
    obj = _load_json(input_path)
    out_obj = run_rag_on_vlm_obj(
        obj,
        device=device,
        tmp_dir=tmp_dir,
        kb=kb,
        emb_model_dir=emb_model_dir,
        expand_local_model_path=expand_local_model_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run RAG on VLM output JSON")
    parser.add_argument("--input", required=True, help="VLM output JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda:0", help="Device for resolve_foodkey, e.g. cuda:0 or cpu")
    parser.add_argument("--tmp_dir", default=str(Path(__file__).resolve().parent / "_rag_tmp"), help="Temp dir")
    parser.add_argument("--kb", default=None, help="Optional KB path for resolve_foodkey")
    parser.add_argument("--emb_model_dir", default=None, help="Override embedding model dir")
    parser.add_argument("--expand_local_model_path", default=None, help="Override local LLM path for expand")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_rag_on_vlm_file(
        in_path,
        out_path,
        device=args.device,
        tmp_dir=tmp_dir,
        kb=args.kb,
        emb_model_dir=args.emb_model_dir,
        expand_local_model_path=args.expand_local_model_path,
    )


if __name__ == "__main__":
    main()
