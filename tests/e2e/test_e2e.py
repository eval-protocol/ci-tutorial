import json
import os
from typing import Any, Dict, List

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow
from eval_protocol.pytest import evaluation_test, AgentRolloutProcessor

from app import draft_eval
from app.prompts import generate_messages


def adapter_examples_to_rows(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    dataset: List[EvaluationRow] = []
    for row in rows:
        dataset.append(EvaluationRow(messages=row["messages"], ground_truth=row.get("ground_truth")))
    return dataset


def _write_dataset_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def to_dict_msgs(msgs):
        return [{"role": m.role, "content": m.content} for m in msgs]

    examples = [
        {"messages": generate_messages("Write a JSON schema validation eval")},
        {"messages": generate_messages("Show how to use num_runs and aggregation")},
        {"messages": generate_messages("Explain rollout processor differences")},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "tasks.jsonl")
_write_dataset_if_missing(DATASET_PATH)


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason="Skipping e2e: FIREWORKS_API_KEY not set")
@evaluation_test(
    input_dataset=[DATASET_PATH],
    dataset_adapter=adapter_examples_to_rows,
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=AgentRolloutProcessor(),
    mode="pointwise",
    num_runs=2,
    passed_threshold=0.6,
    mcp_config_path="tests/integration/mcp_configurations/evalprotocol_mcp.json",
)
def test_eval_author_e2e(row: EvaluationRow) -> EvaluationRow:
    """
    E2E eval: agent processes multiple eval authoring tasks with MCP tools.
    Expects tool usage and eval-protocol knowledge across multiple runs.
    """
    content = row.messages[-1].content or ""
    made_tool_call = any(m.role == "tool" for m in row.messages)
    mentions_ep = "eval-protocol" in content or "evaluation_test" in content

    # Score based on tool usage and EP knowledge
    if made_tool_call and mentions_ep:
        score = 1.0
        reason = "Used tools and mentioned eval-protocol"
    elif made_tool_call or mentions_ep:
        score = 0.7
        reason = "Either used tools or mentioned eval-protocol"
    else:
        score = 0.3
        reason = "No tools used and no EP knowledge shown"

    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
