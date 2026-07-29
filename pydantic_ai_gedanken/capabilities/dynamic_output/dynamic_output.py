"""Dynamic Output capability: let an agent author, verify, and evolve its own output type as Pydantic models."""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.output import StructuredDict
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from ._store import SchemaRecord, SchemaStore
from ._toolset import AUTHOR_TOOL_NAME, DynamicOutputToolset
from ._verification import Verifier, run_verifiers

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.run import AgentRunResult

logger = logging.getLogger(__name__)

_DEFAULT_GUIDANCE = (
    'This agent has no fixed output type: you author it at runtime as a JSON Schema, and it evolves '
    f'gradually across versions. Use `{AUTHOR_TOOL_NAME}(name, json_schema)` to create a schema or evolve '
    'an existing one (the root must be an object schema, ideally with `title`, `description`, `properties` '
    'and `required`). The schema is converted into a Pydantic model — returned to you as Python source — '
    'and that model immediately becomes the output type for your final answer. '
    'After the run completes, verification checks run against the active model; if any fail, you will be '
    're-run with the failure messages — evolve the schema by authoring a new version under the same name. '
    'Prefer small, backward-compatible evolutions over rewrites.'
)


@dataclass
class DynamicOutput(AbstractCapability[AgentDepsT]):
    """Let an agent author its own output type as Pydantic models that evolve gradually over time.

    On each run the capability erases any pre-existing output type (the agent starts with
    output type `None`, until a schema is active in the store) and exposes authoring tools:

    1. the model authors or evolves a schema with `author_output_schema(name, json_schema)`;
       the draft is lint-checked, converted into a Pydantic model by
       [`datamodel-code-generator`](https://koxudaxi.github.io/datamodel-code-generator/),
       stored as a Python module, and activated as the agent's output type — the generated
       source is handed back to the model so it authors against real Python, not raw JSON;
    2. after the run completes, [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]
       runs every [`Verifier`][pydantic_ai_gedanken.capabilities.dynamic_output.Verifier] in
       `verifiers` against the active model; if any fail, the agent is re-run with the
       failure messages appended as context so it can evolve the schema, up to `max_retries`
       times (a warning is emitted if the checks still fail after the last retry).

    Models are persisted per version in `directory` as `<name>.v<version>.py`, indexed by a
    `manifest.json` with statuses (`draft`/`active`/`inactive`), timestamps, and last errors.
    Because each version is a plain module, it can be read and hand-edited on disk; edits are
    picked up on the next run. The active model is served as the output type on every run
    while this capability is attached; to reuse it elsewhere, thread
    `store.active_output_type()` into another agent.

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

    Note: to let the run *end* with the evolved output type, construct the agent with
    [`placeholder_output_type`][pydantic_ai_gedanken.capabilities.dynamic_output.DynamicOutput.placeholder_output_type]
    (its output tool definition is rewritten in place with the active model's schema). Without
    it the agent has no output tool for the capability to rewrite, and authored models only
    take effect on later runs via `store.active_output_type()`.
    """

    directory: Path
    """Directory holding the `<name>.v<version>.py` model modules and the `manifest.json` index."""

    verifiers: list[Verifier] = field(default_factory=list[Verifier])
    """Verification checks run against the active model after each agent run completes.

    Each [`Verifier`][pydantic_ai_gedanken.capabilities.dynamic_output.Verifier] receives the
    generated Pydantic model and returns a
    [`VerificationResult`][pydantic_ai_gedanken.capabilities.dynamic_output.VerificationResult];
    failures are fed back to the agent as a retry run so the model evolves. An empty list
    means only the built-in lint checks (at authoring time) gate the schema."""

    max_retries: int = 3
    """Maximum number of retry runs when model verification fails."""

    parallel: bool = True
    """Whether to run verifiers concurrently via `asyncio.gather`.

    When `True` (the default), all verifiers execute concurrently.
    Set to `False` to run them sequentially in list order."""

    guidance: str | None = None
    """Static system-prompt guidance on the authoring workflow. Cache-stable. Leave `None`
    for the default, or set `''` to omit guidance entirely."""

    _schema_created: bool = field(default=False, init=False, repr=False)
    """Private marker: whether a model has been created since output tools were last erased."""

    _in_retry: bool = field(default=False, init=False, repr=False)
    """When `True`, `wrap_run` skips verification (retry pass-through)."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Opt out of spec-based construction: verifiers are not serializable."""
        return None

    @staticmethod
    def placeholder_output_type(name: str = 'dynamic_output') -> type[JsonSchemaValue]:
        """Placeholder output type whose tool definition is rewritten with the active authored model.

        Args:
            name: Name given to the placeholder structured output.
        """
        return StructuredDict(
            {'type': 'object', 'additionalProperties': True},
            name=name,
            description='Dynamic output; its type is authored at runtime.',
        )

    @property
    def store(self) -> SchemaStore:
        """The disk-backed store. Call `store.active_output_type()` to reuse the evolved model elsewhere."""
        return SchemaStore(self.directory)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on the schema authoring workflow."""
        guidance = _DEFAULT_GUIDANCE if self.guidance is None else self.guidance
        return guidance or None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the schema authoring tools over this capability's store."""
        return DynamicOutputToolset[AgentDepsT](self.store, self)

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Erase any pre-existing output type; serve the active authored model, if any.

        While no authored model is active in the store, every output tool definition is
        dropped — the agent effectively has `output_type=None`. Once a model is active, the
        first output tool definition is rewritten in place with that model's JSON Schema
        (`by_alias=False`, so the model sees the generated field names it can read in the
        module on disk).
        """
        record = self.store.active_record()
        if record is None:
            self._schema_created = False
            return []
        model = self.store.read_model(record)
        return [
            replace(
                tool_def,
                parameters_json_schema=model.model_json_schema(by_alias=False),
                description=inspect.cleandoc(model.__doc__) if model.__doc__ else tool_def.description,
                outer_typed_dict_key=None,
                strict=None,
            )
            for tool_def in tool_defs[:1]
        ]

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Run the agent, then verify the active model. Retry with feedback on failure.

        When the agent is re-run for a retry, this hook fires again on the new run.
        The `_in_retry` flag prevents recursive verification: retry runs pass straight
        through to the handler.
        """
        result = await handler()

        # Retry runs skip verification to avoid infinite recursion.
        if self._in_retry or not self.verifiers:
            return result

        agent = ctx.agent

        for attempt in range(1, self.max_retries + 1):
            failures = await self._verify_active_model()
            if failures is None or not failures:
                return result

            feedback = self._build_feedback(failures, attempt)
            logger.info(
                'Output model verification failed (attempt %d/%d): %s',
                attempt,
                self.max_retries,
                '; '.join(f'{name}: {message}' for name, message in failures),
            )

            if agent is None:  # pragma: no cover — defensive; agent is always set in practice
                warnings.warn(
                    'Output model verification failed but agent is not available on RunContext for retry. '
                    'Returning last result.',
                    stacklevel=2,
                )
                return result

            # Mark that the next run is a retry so wrap_run passes through.
            self._in_retry = True
            try:
                result = await agent.run(feedback, message_history=result.all_messages(), deps=ctx.deps)
            finally:
                self._in_retry = False

        # Final verification after the last retry.
        failures = await self._verify_active_model()
        if failures is None or not failures:
            return result

        warnings.warn(
            f'Output model verification still failing after {self.max_retries} retries: '
            + '; '.join(f'{name}: {message}' for name, message in failures),
            stacklevel=2,
        )
        return result

    async def _verify_active_model(self) -> list[tuple[str, str]] | None:
        """Run the verifiers against the active model, recording the outcome in the store.

        Returns `None` when no model is active (nothing to verify), otherwise the list of
        `(verifier name, message)` failures — empty when every check passed.
        """
        record = self.store.active_record()
        if record is None:
            return None
        model: type[BaseModel] = self.store.read_model(record)
        failures = await run_verifiers(self.verifiers, model, parallel=self.parallel)
        if failures:
            self.store.set_error(
                record.name,
                'Output model verification failed:\n' + '\n'.join(f'- {name}: {msg}' for name, msg in failures),
            )
        else:
            self.store.set_error(record.name, None)
        return failures

    @staticmethod
    def _build_feedback(failures: list[tuple[str, str]], attempt: int) -> str:
        """Build a feedback prompt from output model verification failures."""
        parts = [
            f'Output model verification failed (attempt {attempt}). Evolve the output type with '
            f'`{AUTHOR_TOOL_NAME}` (same name, fixed `json_schema`) to address ONLY the failing checks, '
            f'then produce your final output again.'
        ]
        parts.extend(f'- {name}: {message}' for name, message in failures)
        return '\n'.join(parts)

    # --- SchemaLifecycleHost callbacks (used by DynamicOutputToolset) ---

    def on_schema_authored(self, record: SchemaRecord) -> None:
        """Record that a generated model version is now active."""
        self._schema_created = True
