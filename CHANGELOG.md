# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0]

### Added

- **`DynamicOutput` capability** (`pydantic_ai_gedanken.capabilities.dynamic_output`): lets an agent author, verify, and evolve its own output type as Pydantic models at runtime.
  - `author_output_schema(name, json_schema)` tool: lint-checks a draft JSON Schema, converts it into a Pydantic model via `datamodel-code-generator`, activates it as the agent's output type, and returns the generated Python source to the agent.
  - `list_output_schemas` and `deactivate_output_schema` tools for schema lifecycle management.
  - Post-run verification via user-supplied `Verifier` checks; failures are fed back to the agent as retry runs (configurable `max_retries`, parallel or sequential execution).
  - Disk-backed `SchemaStore`: one Python module per version (`<name>.v<version>.py`) indexed by an atomic `manifest.json` with statuses (`draft`/`active`/`inactive`), timestamps, and last errors. Modules are hand-editable; edits are picked up on the next run.
  - `DynamicOutput.placeholder_output_type()` so the current run can end with the evolved output type, and `store.active_output_type()` to reuse the evolved model in other agents.
