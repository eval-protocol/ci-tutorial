"""
Core prompts for the eval drafting agent.
"""

from typing import Dict


def build_eval_author_system_prompt() -> str:
    return """You are an applied AI engineer whose job is to write tests
called "evals" in the form of code. An "eval" helps determines whether an AI
application is working as expected by programatically assessing the output of
the model. To do this, you will use a library called "eval-protocol" (aka EP)
that helps you easily author, run, and review evals. Evals accept whats called
an EvaluationRow and outputs an EvaluationRow (or multiple EvaluationRows
depending on the mode of the @evaluation_test decorator). In the eval, a score
from 0 to 1 is generated based on the output of the model. Use the provided
tools to help you understand more about eval-protocol so that you can generate
evals for the given task. The tools can help provide examples of evals, guide
you with tutorials on how to use eval-protocol, retrieve reference documentation
for the eval-protocol API, and ask questions about the source code of
eval-protocol.
"""


def build_eval_author_user_prompt(task: str) -> str:
    return (
        "Given the following evaluation task, outline the approach and, if asked, produce code snippets.\n"
        f"TASK: {task}"
    )


def generate_draft_eval_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_eval_author_system_prompt()},
        {"role": "user", "content": build_eval_author_user_prompt(task)},
    ]


def build_extract_code_system_prompt() -> str:
    return """Your job is to extract the final code output from a string. You will be given a string and you will need to strip everything but the final code output from it. Here are some examples:

1. You should return: "print("Hello, world!")" if the input is:
```python
print("Hello, world!")
```
2. You should return: "print("Hello, world!")" if the input is just the code itself like "print("Hello, world!")"
3. You should return: "print("Hello, world!")" if the input is:
Here is the code:

```python
print("Hello, world!")
```
"""


def generate_extract_code_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_extract_code_system_prompt()},
        {"role": "user", "content": text},
    ]
