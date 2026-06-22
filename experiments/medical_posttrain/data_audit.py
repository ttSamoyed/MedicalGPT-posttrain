#!/usr/bin/env python3
"""Audit preference data before DPO/RM/PPO training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    yield {"__error__": f"line {line_no}: {exc}", "__raw__": line[:200]}
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("data", "train", "examples"):
            if isinstance(data.get(key), list):
                yield from data[key]
                return
        yield data
    else:
        yield {"__error__": f"unsupported json root: {type(data).__name__}"}


def text_of(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(str(x) for x in value).strip()
    return ""


def prompt_of(record: dict[str, Any]) -> str:
    direct = text_of(record, ["prompt", "question", "input", "instruction"])
    if direct:
        return direct

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        parts = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role") or ""
            value = turn.get("value") or turn.get("content") or ""
            if role in {"human", "user", "system"} and value:
                parts.append(str(value))
        return "\n".join(parts).strip()
    return ""


def has_bad_format(text: str) -> bool:
    if not text:
        return True
    suspicious = ["答案：答案：", "？？？", "!!!", "<unk>", "null", "None"]
    return any(x in text for x in suspicious)


def looks_generic(text: str) -> bool:
    generic_phrases = [
        "建议咨询医生",
        "请咨询专业医生",
        "无法提供医疗建议",
        "作为人工智能",
        "我不是医生",
    ]
    return any(x in text for x in generic_phrases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON/JSONL preference data path.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = list(iter_json_records(input_path))

    counters: Counter[str] = Counter()
    samples: list[str] = []
    seen_pairs: Counter[tuple[str, str, str]] = Counter()

    for idx, record in enumerate(records):
        if "__error__" in record:
            counters["parse_error"] += 1
            if len(samples) < args.max_examples:
                samples.append(f"- parse error: `{record['__error__']}`")
            continue

        prompt = prompt_of(record)
        chosen = text_of(record, ["chosen", "response_chosen", "answer_chosen"])
        rejected = text_of(record, ["rejected", "response_rejected", "answer_rejected"])

        if not prompt:
            counters["missing_prompt"] += 1
        if not chosen:
            counters["missing_chosen"] += 1
        if not rejected:
            counters["missing_rejected"] += 1
        if chosen and rejected and chosen == rejected:
            counters["same_chosen_rejected"] += 1
        if has_bad_format(chosen):
            counters["chosen_bad_format"] += 1
        if has_bad_format(rejected):
            counters["rejected_bad_format"] += 1
        if looks_generic(chosen):
            counters["chosen_generic_disclaimer"] += 1
        if looks_generic(rejected):
            counters["rejected_generic_disclaimer"] += 1
        if len(chosen) < len(rejected) * 0.35 and len(rejected) > 80:
            counters["chosen_much_shorter"] += 1

        pair_key = (prompt[:200], chosen[:200], rejected[:200])
        seen_pairs[pair_key] += 1

        if len(samples) < args.max_examples and (
            chosen == rejected
            or has_bad_format(chosen)
            or looks_generic(chosen)
            or len(chosen) < len(rejected) * 0.35
        ):
            samples.append(
                "\n".join(
                    [
                        f"- sample #{idx}",
                        f"  - prompt: {prompt[:120]}",
                        f"  - chosen: {chosen[:160]}",
                        f"  - rejected: {rejected[:160]}",
                    ]
                )
            )

    duplicate_count = sum(count - 1 for count in seen_pairs.values() if count > 1)

    lines = [
        "# Preference Data Audit",
        "",
        f"- input: `{input_path}`",
        f"- total records: {len(records)}",
        f"- duplicate pairs: {duplicate_count}",
        "",
        "## Issue Counts",
        "",
    ]

    if counters:
        for key, value in counters.most_common():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- no obvious issues found by heuristic checks")

    lines.extend(["", "## Suspicious Samples", ""])
    lines.extend(samples or ["- no suspicious sample captured"])
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
