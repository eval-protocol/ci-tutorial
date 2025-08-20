# ci-tutorial

Best practices for running CI pipelines for AI applications with eval-protocol using GitHub Actions.

## Overview

This example demonstrates a specialized eval-protocol agent that writes evaluation tests:

- **App layer**: `app/prompts.py` builds messages for an eval authoring expert; `app/api.py` exposes a public API `draft_eval(task: str) -> str` which returns assistant text for an eval-drafting response via LiteLLM.
- **Shared usage**: Evals import the same prompts/helpers used by the app.
- **Unit/Integration/E2E** separation via `pytest` markers following eval-protocol best practices.

### Public API

- `app.draft_eval(task: str) -> str`: Generates an eval-drafting response using LiteLLM. Uses `LITELLM_MODEL` and provider API key(s) from the environment; defaults to `fireworks_ai/accounts/fireworks/models/gpt-oss-120b`.
- `app.prompts.generate_messages(task: str) -> list[dict[str, str]]`: Returns the `[system, user]` messages used across tests.
- Optional adapters: `make_openai_chat_completion_callable`, `make_litellm_chat_completion_callable`.

## Local setup

```bash
uv pip install -e .[test]
pytest -q -m unit
FIREWORKS_API_KEY=... pytest -q -m integration
FIREWORKS_API_KEY=... pytest -q -m e2e
```

Requirements:
- Python 3.11+
- A provider key supported by LiteLLM (examples below use Fireworks via `FIREWORKS_API_KEY`).

### Environment variables
- `LITELLM_MODEL`: LiteLLM model identifier. Defaults to `fireworks_ai/accounts/fireworks/models/gpt-oss-120b`.
- Provider API key(s) as required by LiteLLM, e.g. `FIREWORKS_API_KEY` for Fireworks.
- Optional (used in E2E workflow): `EP_PRINT_SUMMARY=1`, `EP_SUMMARY_JSON=artifacts/summaries/`.

## Usage

Call the high-level API; it uses LiteLLM under the hood. Set `LITELLM_MODEL` and the corresponding provider key.

```python
from app import draft_eval

# export LITELLM_MODEL="fireworks_ai/accounts/fireworks/models/gpt-oss-120b"  # default
# export FIREWORKS_API_KEY=...

code = draft_eval("validate JSON schema of an LLM response")
print(code)
```

## GitHub Actions

- **Unit on PR and push**: `.github/workflows/ci.yml` runs `pytest -m unit` on pull requests and pushes to `main/trunk/master`. Requires `FIREWORKS_API_KEY` secret.
- **Integration on push and daily (12:00 UTC)**: `.github/workflows/integration.yml` runs `pytest -m integration` on pushes to `main/trunk/master` and on a daily schedule.
- **Nightly E2E (08:00 UTC)**: `.github/workflows/nightly-e2e.yml` runs `pytest -m e2e`, with summaries uploaded as artifacts.

All workflows use `uv` to create a virtualenv and install `.[test]`.

## Test Categories

- **Unit**: Single-turn eval authoring responses (frequent, reliable)
- **Integration**: Agent rollout with MCP tools from `https://evalprotocol.io/mcp` (semi-frequent, tool-dependent)
- **E2E**: Dataset of eval authoring tasks with multiple runs (infrequent, comprehensive)

Notes:
- Tests are skipped if `FIREWORKS_API_KEY` is not set.
- Example datasets are auto-created under `tests/**/data/tasks.jsonl` if missing.
