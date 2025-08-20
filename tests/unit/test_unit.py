import os
import pytest

from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

from app.prompts import generate_messages

skip_reason = "Skipping unit test: FIREWORKS_API_KEY not set"


@pytest.mark.unit
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason=skip_reason)
@evaluation_test(
    input_messages=[
        generate_messages(task="validate JSON schema of an LLM response"),
        generate_messages(task="evaluate code generation quality"),
    ],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.8,  # Expect high reliability for single-turn responses
)
def test_eval_author_unit(row: EvaluationRow) -> EvaluationRow:
    """
    Unit eval for the eval authoring agent: single-turn model call to check if it references
    eval-protocol constructs.
    """
    assistant_response = row.messages[-1].content or ""
    score = 0.0
    reason = "Response does not mention key eval-protocol constructs."

    # Check for keywords indicating knowledge of eval-protocol
    if "@evaluation_test" in assistant_response and "score" in assistant_response:
        score = 1.0
        reason = "Response mentions @evaluation_test and score."
    elif "@evaluation_test" in assistant_response:
        score = 0.5
        reason = "Response mentions @evaluation_test but not score."

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
        metrics={"keyword_check": MetricResult(score=score, reason=reason, is_score_valid=True)},
    )
    return row
