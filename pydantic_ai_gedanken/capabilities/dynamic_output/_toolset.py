"""Model-facing tools for authoring and managing dynamic output schemas."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from ._codegen import ModelGenerationError
from ._lint import lint_json_schema
from ._store import SchemaRecord, SchemaStore

AUTHOR_TOOL_NAME = 'author_output_schema'
"""Name of the tool that creates or evolves the active JSON Schema."""
LIST_TOOL_NAME = 'list_output_schemas'
"""Name of the tool that lists the schema manifest."""
DEACTIVATE_TOOL_NAME = 'deactivate_output_schema'
"""Name of the tool that deactivates a schema."""


class SchemaLifecycleHost(Protocol):
    """Callbacks the toolset uses to report schema lifecycle transitions to its capability."""

    def on_schema_authored(self, record: SchemaRecord) -> None:
        """Notify that a schema version was authored and activated."""
        ...


class DynamicOutputToolset(FunctionToolset[AgentDepsT]):
    """Toolset exposing the schema authoring and management tools over a `SchemaStore`."""

    def __init__(self, store: SchemaStore, host: SchemaLifecycleHost) -> None:
        """Build the toolset.

        Args:
            store: The disk-backed schema store to operate on.
            host: The owning capability, notified of lifecycle transitions.
        """
        super().__init__()
        self._store = store
        self._host = host
        self.add_function(self.author_output_schema, name=AUTHOR_TOOL_NAME)
        self.add_function(self.list_output_schemas, name=LIST_TOOL_NAME)
        self.add_function(self.deactivate_output_schema, name=DEACTIVATE_TOOL_NAME)

    async def author_output_schema(self, name: str, json_schema: dict[str, Any]) -> str:
        """Create a new output JSON Schema, or evolve an existing one, and activate it.

        The schema is lint-checked and converted into a Pydantic model, which is stored as a
        new version and immediately becomes the agent's active output type; the generated
        model is returned so you can see the exact type you must produce. Verification checks
        run against it after the run completes; failures come back as feedback so the model
        can be evolved.

        Args:
            name: Schema name (lowercase letters, digits and underscores; must start with a letter).
                Reusing an existing name evolves that schema to a new version.
            json_schema: The JSON Schema document. The root must be an object schema
                (`"type": "object"`), typically with `title`, `description`, `properties`
                and `required`.
        """
        lint = lint_json_schema(json_schema)
        if not lint.ok:
            raise ModelRetry(
                'JSON Schema failed format/lint checks:\n- '
                + '\n- '.join(lint.errors)
                + '\nFix the schema and call the tool again.'
            )
        try:
            record = self._store.write_draft(name, json_schema)
        except (ValueError, ModelGenerationError) as exc:
            raise ModelRetry(str(exc)) from exc
        activated = self._store.activate(name)
        self._host.on_schema_authored(activated)
        message = (
            f'Version {record.version} of output schema {name!r} was converted into the Pydantic model '
            f'`{record.class_name}` and is now the active output type. Produce your final output '
            f'conforming to it:\n\n```python\n{self._store.read_source(record)}```'
        )
        if lint.warnings:
            message += '\nLint warnings (non-blocking):\n- ' + '\n- '.join(lint.warnings)
        return message

    async def list_output_schemas(self) -> str:
        """List every authored output schema with its version, generated model, status, and timestamps."""
        records = self._store.list_all()
        if not records:
            return 'No output schemas have been authored yet.'
        return json.dumps([record.model_dump() for record in records], indent=2)

    async def deactivate_output_schema(self, name: str) -> str:
        """Deactivate an output schema so it is no longer used as the output type.

        Args:
            name: Name of the schema to deactivate.
        """
        if self._store.deactivate(name):
            return f'Output schema {name!r} is now inactive.'
        return f'No output schema named {name!r} exists.'

    def read_source(self, record: SchemaRecord) -> str:
        """Read the source of the generated Pydantic model backing a manifest record."""
        return self._store.read_source(record)
