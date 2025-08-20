import os
import json
import pytest
from typing import List, Dict, Any

from eval_protocol.models import EvaluationRow, EvaluateResult, Message, MetricResult
from eval_protocol.pytest import evaluation_test, AgentRolloutProcessor
from eval_protocol.dataset_logger import default_logger

from app.prompts import generate_draft_eval_messages

skip_reason = "Skipping integration test: FIREWORKS_API_KEY not set"


def _write_integration_dataset_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    examples = [
        {"task": "validate JSON schema of an LLM response"},
        {"task": "evaluate code generation quality"},
        {"task": "write an eval for hallucination detection"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "tasks.jsonl")
_write_integration_dataset_if_missing(DATASET_PATH)


def eval_author_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapts a simple list of tasks into EvaluationRow objects for the eval authoring agent.
    """
    dataset: List[EvaluationRow] = []
    for row_data in rows:
        task = row_data["task"]
        messages = generate_draft_eval_messages(task=task)
        ep_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        dataset.append(EvaluationRow(messages=ep_messages))
    return dataset


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason=skip_reason)
@evaluation_test(
    input_dataset=[DATASET_PATH],
    dataset_adapter=eval_author_dataset_adapter,
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=AgentRolloutProcessor(),
    mode="batch",  # Agent rollouts are typically batch mode
    passed_threshold=0.7,  # Expect reasonable success with tool calls
    mcp_config_path="tests/integration/mcp_configurations/evalprotocol_mcp.json",
)
def test_eval_author_integration(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Integration eval for the eval authoring agent: uses agent rollout with MCP tool-calling.
    Checks if a tool call occurred or if the response references eval-protocol documentation.
    """
    for row in rows:
        score = 0.0
        reason = "No tool calls made and no relevant documentation referenced."
        tool_call_made = False
        doc_referenced = False

        for message in row.messages:
            if message.role == "tool":
                tool_call_made = True
                break
            if message.content and (
                "eval-protocol.io" in message.content or "evalprotocol.io/docs" in message.content
            ):
                doc_referenced = True

        if tool_call_made:
            score = 1.0
            reason = "Agent made at least one tool call."
        elif doc_referenced:
            score = 0.8
            reason = "Agent referenced eval-protocol documentation."

        row.evaluation_result = EvaluateResult(
            score=score,
            reason=reason,
            metrics={
                "tool_call_check": MetricResult(
                    score=1.0 if tool_call_made else 0.0,
                    reason="Tool call made" if tool_call_made else "No tool call",
                    is_score_valid=True,
                ),
                "doc_reference_check": MetricResult(
                    score=1.0 if doc_referenced else 0.0,
                    reason="Doc referenced" if doc_referenced else "No doc reference",
                    is_score_valid=True,
                ),
            },
        )
        default_logger.log(row)  # Log the row after evaluation
    return rows
