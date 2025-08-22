import os
import pytest
from typing import List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import evaluation_test, AgentRolloutProcessor
from eval_protocol.dataset_logger import default_logger

from app.prompts import generate_draft_eval_messages

skip_reason = "Skipping integration test: FIREWORKS_API_KEY not set"

EXAMPLES = [
    {"task": "write an eval that checks if the JSON schema of an LLM response is correct"},
    {"task": "write an eval that checks if the code generation quality is good"},
    {"task": "write an eval that checks if the hallucination detection is good"},
]


def generate_rows() -> List[EvaluationRow]:
    """
    Build input_rows directly from the JSONL dataset for the eval authoring agent.
    """
    rows: List[EvaluationRow] = []
    for row_data in EXAMPLES:
        task = row_data["task"]
        messages = generate_draft_eval_messages(task=task)
        ep_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        rows.append(EvaluationRow(messages=ep_messages))
    return rows


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason=skip_reason)
@pytest.mark.skipif(bool(os.getenv("CI")), reason="Skipping in CI since this is just for manual eval development")
@evaluation_test(
    input_rows=generate_rows(),
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"}],
    rollout_processor=AgentRolloutProcessor(),
    mode="pointwise",  # Agent rollouts are typically batch mode
    num_runs=2,
    mcp_config_path="app/evalprotocol_mcp.json",
)
def test_run_rollouts(row: EvaluationRow) -> EvaluationRow:
    """
    For running rollouts for manual eval development.
    """
    return row
