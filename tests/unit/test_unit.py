import json
import os
from typing import Any, Dict, List
from eval_protocol import Message
import pytest

from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

from app.prompts import generate_extract_code_messages

skip_reason = "Skipping unit test: FIREWORKS_API_KEY not set"

GROUND_TRUTH = "print('Hello, world!')"

TEXT_WITH_CODE_1 = f"""
```python
{GROUND_TRUTH}
```
"""

TEXT_WITH_CODE_2 = GROUND_TRUTH

TEXT_WITH_CODE_3 = f"""
```
{GROUND_TRUTH}
```
"""

TEXT_WITH_CODE_4 = f"""
Here is the code:

```python
{GROUND_TRUTH}
```
"""


def _write_integration_dataset_if_missing(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    examples = [
        {"text": TEXT_WITH_CODE_1, "ground_truth": GROUND_TRUTH},
        {"text": TEXT_WITH_CODE_2, "ground_truth": GROUND_TRUTH},
        {"text": TEXT_WITH_CODE_3, "ground_truth": GROUND_TRUTH},
        {"text": TEXT_WITH_CODE_4, "ground_truth": GROUND_TRUTH},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "extract_code_tasks.jsonl")
_write_integration_dataset_if_missing(DATASET_PATH)


def extract_code_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """
    Adapts a simple list of tasks into EvaluationRow objects for the eval authoring agent.
    """
    dataset: List[EvaluationRow] = []
    for row_data in rows:
        text = row_data["text"]
        ground_truth = row_data["ground_truth"]
        messages = generate_extract_code_messages(text=text)
        ep_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
        dataset.append(EvaluationRow(messages=ep_messages, ground_truth=ground_truth))
    return dataset


@pytest.mark.unit
@pytest.mark.skipif(not os.getenv("FIREWORKS_API_KEY"), reason=skip_reason)
@evaluation_test(
    input_dataset=[DATASET_PATH],
    dataset_adapter=extract_code_dataset_adapter,
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    passed_threshold=1.0,  # Expect high reliability for single-turn responses
)
def test_eval_author_unit(row: EvaluationRow) -> EvaluationRow:
    """
    Unit eval for the eval authoring agent: single-turn model call to check if it references
    eval-protocol constructs.
    """
    if row.messages[-1].content == row.ground_truth:
        score = 1.0
        reason = "Response contains the correct code."
    else:
        score = 0.0
        reason = "Response does not contain the correct code."

    row.evaluation_result = EvaluateResult(
        score=score,
        reason=reason,
        metrics={"keyword_check": MetricResult(score=score, reason=reason, is_score_valid=True)},
    )
    return row
