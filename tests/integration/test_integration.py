import os
import pytest
from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test, AgentRolloutProcessor
from eval_protocol.dataset_logger import default_logger

from app.prompts import generate_draft_eval_messages

skip_reason = "Skipping integration test: FIREWORKS_API_KEY not set"


def generate_rows() -> List[EvaluationRow]:
    """
    Build input_rows directly from the JSONL dataset for the eval authoring agent.
    """
    examples = [
        {"task": "write an eval that checks if the JSON schema of an LLM response is correct"},
        {"task": "write an eval that checks if the code generation quality is good"},
        {"task": "write an eval that checks if the hallucination detection is good"},
    ]
    rows: List[EvaluationRow] = []
    for row_data in examples:
        task = row_data["task"]
        messages = generate_draft_eval_messages(task=task)
        ep_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        rows.append(EvaluationRow(messages=ep_messages))
    return rows


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason=skip_reason)
@evaluation_test(
    input_rows=generate_rows(),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct"}],
    rollout_processor=AgentRolloutProcessor(),
    mode="pointwise",  # Agent rollouts are typically batch mode
    num_runs=5,
    mcp_config_path="tests/integration/mcp_configurations/evalprotocol_mcp.json",
)
def test_eval_author_integration(row: EvaluationRow) -> EvaluationRow:
    """
    Integration eval for the eval authoring agent: uses agent rollout with MCP tool-calling.
    Checks if a tool call occurred or if the response references eval-protocol documentation.
    """
    return row
