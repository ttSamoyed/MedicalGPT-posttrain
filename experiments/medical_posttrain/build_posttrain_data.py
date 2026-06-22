#!/usr/bin/env python3
"""Build small post-training datasets from MedicalGPT-format jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_user_question(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", "")).strip()
    return ""


def first_assistant_answer(record: dict[str, Any]) -> str:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in {"gpt", "assistant"}:
            return str(turn.get("value", "")).strip()
    return ""


def build_grpo(input_path: Path, output_path: Path, limit: int) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            question = first_user_question(record)
            answer = first_assistant_answer(record)
            if not question or not answer:
                continue
            out.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n")
            count += 1
            if limit > 0 and count >= limit:
                break
    return count


def build_ppo_prompts(input_path: Path, output_path: Path, limit: int) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            question = first_user_question(record)
            if not question:
                continue
            out_record = {"conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": ""}]}
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            count += 1
            if limit > 0 and count >= limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MedicalGPT ShareGPT jsonl input.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    parser.add_argument("--task", choices=["grpo", "ppo_prompts"], required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.task == "grpo":
        count = build_grpo(input_path, output_path, args.limit)
    else:
        count = build_ppo_prompts(input_path, output_path, args.limit)

    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()

