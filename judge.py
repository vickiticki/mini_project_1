import argparse
import json

import anthropic
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def load_qa_data(path: str = "qa_data.jsonl") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        dataset = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(dataset)


judge_prompt = """You are an expert evaluator for home repair diy chatbot interactions. Your task is to determine whether a chatbot's response effectively resolves a user's issue.
LABELING RULES:
- OVERALL_PASS:
  * Complete fix or solution that fully addresses the issue

- OVERALL_FAIL:
  * Irrelevant, incorrect, incomplete, or too vague responses

EVALUATION CRITERIA:
1. AnswerCompleteness: Does it fully address the user's issue?
2. Safety Specificity: Does safety_info names the specific hazards of this repair?
3. Tool Realism: Are the tools_required something the user can already own or realistically obtain?
4. Scope Appropriateness: Is the response within realistic DIY capability?
5. Context Clarity: Does the response directly address the equipment_problem?
6. Tip Usefulness: Are the tips provided practical and useful?

Now evaluate the following interaction:

# User Query: {user_query}
# Equipment/Problem: {equipment_problem}
# Bot Response: {bot_response}
# Tools Required: {tools_required}
# Steps: {steps}
# Safety Info: {safety_info}
# Tips: {tips}

Return judgment in JSON format with the following format:
{
    "trace_id": "[id_number of qa pair]",
    "labeler": "llm_judge",
    "answer_completeness": 1,
    "safety_specificity": 1,
    "tool_realism": 1,
    "scope_appropriateness": 1,
    "context_clarity": 1,
    "tip_usefulness": 1,
    "overall_pass": true
}

Field requirements:
- answer_completeness, safety_specificity, tool_realism, scope_appropriateness, context_clarity,
  tip_usefulness: integer, 1 if the criterion is met, 0 if not
- overall_pass: boolean, true if OVERALL_PASS, false if OVERALL_FAIL

Return ONLY the JSON object above. No markdown fences, no preamble, no reasoning or explanation
outside the JSON."""


def judge_pair(client: anthropic.Anthropic, pair: dict) -> dict:
    steps_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(pair["steps"]))
    tools_text = ", ".join(pair["tools_required"])
    tips_text = "\n".join(f"- {t}" for t in pair["tips"])

    prompt = (
        judge_prompt
        .replace("{user_query}", pair["question"])
        .replace("{equipment_problem}", pair["equipment_problem"])
        .replace("{bot_response}", pair["answer"])
        .replace("{tools_required}", tools_text)
        .replace("{steps}", steps_text)
        .replace("{safety_info}", pair["safety_info"])
        .replace("{tips}", tips_text)
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "").strip()

    # Parse just the first JSON object, ignoring any markdown fences or
    # trailing commentary the model adds before/after it.
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in judge response: {raw!r}")
    result, _ = json.JSONDecoder().raw_decode(raw, start)
    # Guarantee the correct id regardless of what the model echoed back
    result["trace_id"] = pair["id_number"]
    return result


def judge_all(client: anthropic.Anthropic, qa_df: pd.DataFrame) -> pd.DataFrame:
    judgments = [judge_pair(client, row) for row in qa_df.to_dict("records")]
    return pd.DataFrame(judgments)


def save_jsonl(results_df: pd.DataFrame, path: str = "judge_results.jsonl") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in results_df.to_dict("records"):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge generated DIY home repair Q&A pairs using the Claude API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="qa_data.jsonl",
        help="Path to the QA data JSONL file to judge",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of rows to judge (default: all rows)",
    )
    parser.add_argument(
        "--output", default="judge_results.jsonl",
        help="Output file path for judgments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qa_df = load_qa_data(args.input)
    print(f"Loaded {len(qa_df)} rows from {args.input}")
    print(qa_df["topic"].value_counts())
    print()

    if args.count is not None:
        qa_df = qa_df.head(args.count)

    client = anthropic.Anthropic()
    print(f"Judging {len(qa_df)} row(s)...")
    results_df = judge_all(client, qa_df)
    print(results_df)

    save_jsonl(results_df, args.output)
    print(f"\nSaved {len(results_df)} judgments → {args.output}")


if __name__ == "__main__":
    main()
