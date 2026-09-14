#!/usr/bin/env python3
"""Interactive CLI for manually reading through QA pairs and scoring them against
the same rubric used by judge.py's LLM judge."""

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually review and label DIY home repair Q&A pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="qa_data.jsonl",
        help="Path to the QA data JSONL file to review",
    )
    parser.add_argument(
        "--output", default="human_labels.jsonl",
        help="Output file path for human labels (appended to, resumable)",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_labeled_ids(path: str) -> set:
    try:
        return {row["trace_id"] for row in load_jsonl(path)}
    except FileNotFoundError:
        return set()


def print_pair(pair: dict, position: int, total: int) -> None:
    print("=" * 70)
    print(f"[{position}/{total}] id_number={pair['id_number']}  topic={pair['topic']}")
    print("=" * 70)
    print(f"\nQuestion:\n{pair['question']}")
    print(f"\nEquipment/Problem:\n{pair['equipment_problem']}")
    print(f"\nAnswer:\n{pair['answer']}")
    print("\nTools Required:")
    for tool in pair["tools_required"]:
        print(f"  - {tool}")
    print("\nSteps:")
    for i, step in enumerate(pair["steps"], 1):
        print(f"  {i}. {step}")
    print(f"\nSafety Info:\n{pair['safety_info']}")
    print("\nTips:")
    for tip in pair["tips"]:
        print(f"  - {tip}")
    print()


# (field, prompt) pairs, in the same order and with the same meaning as judge.py's rubric
CRITERIA = [
    ("answer_completeness", "Does it fully address the user's issue?"),
    ("safety_specificity", "Does safety_info name the specific hazards of this repair?"),
    ("tool_realism", "Are the tools_required something the user can already own or realistically obtain?"),
    ("scope_appropriateness", "Is the response within realistic DIY capability?"),
    ("context_clarity", "Does the response directly address the equipment_problem?"),
    ("tip_usefulness", "Are the tips provided practical and useful?"),
]


class QuitReview(Exception):
    pass


def prompt_start() -> str:
    while True:
        choice = input("Review this item? [c]ontinue / [s]kip / [q]uit: ").strip().lower()
        if choice in ("c", "continue"):
            return "continue"
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("q", "quit"):
            return "quit"
        print("Please enter c, s, or q.")


def prompt_binary(question: str) -> int:
    while True:
        choice = input(f"{question} [y/n/q]: ").strip().lower()
        if choice in ("y", "yes", "1"):
            return 1
        if choice in ("n", "no", "0"):
            return 0
        if choice in ("q", "quit"):
            raise QuitReview
        print("Please enter y, n, or q.")


def prompt_overall_pass() -> bool:
    while True:
        choice = input("Overall pass? [y/n/q]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        if choice in ("q", "quit"):
            raise QuitReview
        print("Please enter y, n, or q.")


def main() -> None:
    args = parse_args()

    pairs = load_jsonl(args.input)
    labeled_ids = load_labeled_ids(args.output)
    remaining = [p for p in pairs if p["id_number"] not in labeled_ids]

    print(f"Loaded {len(pairs)} pairs from {args.input}")
    if labeled_ids:
        print(f"{len(labeled_ids)} already labeled in {args.output} — skipping those")
    print(f"{len(remaining)} pair(s) to review\n")

    if not remaining:
        print("Nothing left to review.")
        return

    reviewed_this_session = 0
    with open(args.output, "a", encoding="utf-8") as out_fh:
        for i, pair in enumerate(remaining, 1):
            print_pair(pair, i, len(remaining))
            start_choice = prompt_start()

            if start_choice == "quit":
                break
            if start_choice == "skip":
                print()
                continue

            try:
                record = {
                    "trace_id": pair["id_number"],
                    "labeler": "human",
                }
                for field, question in CRITERIA:
                    record[field] = prompt_binary(question)
                record["overall_pass"] = prompt_overall_pass()
            except QuitReview:
                break

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            reviewed_this_session += 1
            print()

    print(f"Session complete — labeled {reviewed_this_session} pair(s). Saved to {args.output}")


if __name__ == "__main__":
    main()
