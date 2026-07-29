# pydantic-ai-gedanken

Experimental capabilities for [Pydantic AI](https://ai.pydantic.dev/) agents.

## DynamicOutput

`DynamicOutput` is an agent capability that removes the need to fix an output type up front: the agent authors its own output type at runtime as a JSON Schema, which is converted into a real Pydantic model that evolves gradually across versions.

How it works:

1. **Author** — the agent calls `author_output_schema(name, json_schema)`. The draft schema is lint-checked, converted into a Pydantic model by [datamodel-code-generator](https://koxudaxi.github.io/datamodel-code-generator/), and the generated Python source is handed back to the agent so it authors against real code, not raw JSON.
2. **Activate** — the generated model immediately becomes the agent's output type for its final answer. Companion tools `list_output_schemas` and `deactivate_output_schema` manage the lifecycle.
3. **Verify** — after the run completes, user-supplied `Verifier` checks run against the active model. If any fail, the agent is re-run with the failure messages so it can evolve the schema, up to `max_retries` times.
4. **Persist** — every version is stored on disk as a plain Python module (`<name>.v<version>.py`) indexed by a `manifest.json` with statuses (`draft`/`active`/`inactive`), timestamps, and last errors. Modules can be read and hand-edited; edits are picked up on the next run.

### Usage

```python
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_gedanken.capabilities.dynamic_output import (
    DynamicOutput,
    VerificationResult,
    Verifier,
)


async def requires_name(model: type[BaseModel]) -> VerificationResult:
    ok = 'name' in model.model_fields
    return VerificationResult(passed=ok, message='ok' if ok else "declare a 'name' field")


dynamic_output = DynamicOutput(
    directory=Path('.output_models'),
    verifiers=[Verifier(name='requires_name', check_fn=requires_name)],
)
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[dynamic_output],
    output_type=[str, DynamicOutput.placeholder_output_type()],
)
```

To let the run *end* with the evolved output type, construct the agent with `DynamicOutput.placeholder_output_type()` — its output tool definition is rewritten in place with the active model's schema. To reuse the evolved model in another agent, thread `dynamic_output.store.active_output_type()` into it.

## Development

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
make install    # install package + dev deps and set up pre-commit hooks
make format     # auto-format and fix with ruff
make lint       # check formatting and lint
make typecheck  # run pyright
make test       # run pytest
make            # run all of the above
```
