import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def load_qa_data(path: str = "qa_data.jsonl") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        dataset = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(dataset)


if __name__ == "__main__":
    qa_df = load_qa_data()
    print(f"Loaded {len(qa_df)} rows from qa_data.jsonl")
    print(qa_df["topic"].value_counts())

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
#Bot Response: {bot_response}

Return judgment in JSON format with the following format:
{
    "trace_id": "[id_number of qa pair]",
    "labeler": "llm_evaluator",
    "answer_completeness": "<1 if complete, 0 if incomplete>",
    "safety_specificity": "<1 if specific, 0 if not specific>",
    "tool_realism": "<1 if realistic, 0 if not realistic>",
    "scope_appropriateness": "<1 if appropriate, 0 if not appropriate>",
    "context_clarity": "<1 if clear, 0 if not clear>",
    "tip_usefulness": "<1 if useful, 0 if not useful>",
    "overall_judgment": "<true if OVERALL_PASS, false if OVERALL_FAIL>"
}
"""
