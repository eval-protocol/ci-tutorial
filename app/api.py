import os
from typing import Any, Callable, Dict, List

from eval_protocol.models import EvaluationRow, Message
from eval_protocol.pytest import AgentRolloutProcessor, RolloutProcessorConfig
import litellm

from .prompts import generate_draft_eval_messages, generate_extract_code_messages


def _extract_assistant_content(result: Any) -> str:
    """
    Best-effort extraction of assistant content from common client return types.
    Supports:
    - str (already content)
    - OpenAI responses: obj.choices[0].message.content or dict variant
    - LiteLLM responses: same as OpenAI
    """
    if isinstance(result, str):
        return result

    # dict-style
    if isinstance(result, dict):
        try:
            return result["choices"][0]["message"].get("content") or result["choices"][0].get("text") or ""
        except Exception:
            pass

    # attribute-style (OpenAI SDK objects)
    try:
        choices = getattr(result, "choices")
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if content is not None:
                    return content
            # some providers use .text
            text = getattr(choices[0], "text", None)
            if isinstance(text, str):
                return text
    except Exception:
        pass

    raise ValueError("Unable to extract assistant content from completion result")


def draft_eval(task: str, model: str = "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct") -> str:
    """
    Public API: Given a task, return code for an eval as a string.

    Uses environment variables to choose a default provider at runtime:
    - LITELLM_MODEL (preferred): if set, uses LiteLLM with the given model.
      Works with many providers (e.g., set FIREWORKS_API_KEY for Fireworks).
    - Otherwise, if OPENAI_API_KEY is set and `openai` is installed, uses OpenAI
      with model from OPENAI_MODEL (default: "gpt-4o-mini").

    Optional envs:
    - LITELLM_TEMPERATURE or OPENAI_TEMPERATURE to control sampling (default 0).

    Returns assistant text content. Raise a clear error if no provider is configured.
    """
    messages = generate_draft_eval_messages(task)
    mcp_config_path = os.path.join(os.path.dirname(__file__), "evalprotocol_mcp.json")
    config = RolloutProcessorConfig(
        completion_params={"model": model},
        mcp_config_path=mcp_config_path,
    )
    ep_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
    agent_rollout_processor = AgentRolloutProcessor()
    tasks = agent_rollout_processor([EvaluationRow(messages=ep_messages)], config)
    result = tasks[0].result()
    content = result.messages[-1].content
    if isinstance(content, list):
        text = content[0].text
        return text
    elif isinstance(content, str):
        return content
    raise ValueError(f"Content is not a string: {content}")


def extract_code(text: str) -> str:
    """
    Extract code from a string.
    """
    messages = generate_extract_code_messages(text)
    litellm_model = os.getenv("LITELLM_MODEL", "fireworks_ai/accounts/fireworks/models/gpt-oss-120b")
    result = litellm.completion(model=litellm_model, messages=messages)
    return _extract_assistant_content(result)


def make_openai_chat_completion_callable(client: Any, model: str) -> Callable[..., Any]:
    """
    Convenience adapter for OpenAI-compatible clients.

    Usage:
        from openai import OpenAI
        client = OpenAI()
        run = make_openai_chat_completion_callable(client, model="gpt-4o-mini")
        text = draft_eval("task...")
    """

    def _run(messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        # Avoid strict typing on the client to keep openai optional
        return client.chat.completions.create(model=model, messages=messages, **kwargs)

    return _run


def make_litellm_chat_completion_callable(model: str) -> Callable[..., Any]:
    """
    Convenience adapter for LiteLLM. Requires `litellm` to be installed.
    """

    def _run(messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        import litellm  # local import to avoid hard dependency at import time

        return litellm.completion(model=model, messages=messages, **kwargs)

    return _run
