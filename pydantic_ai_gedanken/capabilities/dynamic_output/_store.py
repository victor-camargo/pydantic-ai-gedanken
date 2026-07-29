"""Disk-backed store for agent-authored output types, indexed by a `manifest.json`.

The agent authors a JSON Schema; each version is converted into a Pydantic model and
persisted as one Python module per version (`<name>.v<version>.py`) next to a
`manifest.json` describing every schema ever created: its current version, the class it
generated, status (`draft` / `active` / `inactive`), timestamps, and the last validation error.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import JsonSchemaValue

from ._codegen import generate_model_source, load_model, model_class_name

_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')

SchemaStatus = Literal['draft', 'active', 'inactive']
"""Lifecycle status of an authored schema.

- `'draft'`: authored but not yet validated and approved.
- `'active'`: validated, approved, and currently serving as the agent's output type.
- `'inactive'`: previously active (or explicitly deactivated); kept for history.
"""


class SchemaRecord(BaseModel):
    """Manifest entry describing one authored output type."""

    name: str
    """Unique schema name (`^[a-z][a-z0-9_]*$`)."""
    source_file: str
    """File name of the latest generated Pydantic model module, relative to the store directory."""
    class_name: str
    """Name of the root model class defined in `source_file`."""
    version: int = 1
    """Monotonically increasing version, bumped on each evolution of the schema."""
    status: SchemaStatus = 'draft'
    """Current lifecycle status."""
    created_at: str
    """ISO-8601 UTC timestamp of the first version of this schema."""
    updated_at: str
    """ISO-8601 UTC timestamp of the latest evolution."""
    activated_at: str | None = None
    """ISO-8601 UTC timestamp of the latest activation, or `None` if never activated."""
    last_error: str | None = None
    """The most recent lint/evaluation error for this schema, or `None`."""


class _Manifest(BaseModel):
    schemas: list[SchemaRecord] = []


def _utcnow() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass
class SchemaStore:
    """Disk-backed store of agent-authored Pydantic models with an atomic `manifest.json` index."""

    directory: Path
    """Directory holding the `<name>.v<version>.py` model modules and the `manifest.json` index."""

    @property
    def _manifest_path(self) -> Path:
        return self.directory / 'manifest.json'

    def _load_manifest(self) -> _Manifest:
        try:
            return _Manifest.model_validate_json(self._manifest_path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            # Missing or corrupt manifest reads as "no schemas" rather than crashing.
            return _Manifest()

    def _save_manifest(self, manifest: _Manifest) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix='manifest.', suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(manifest.model_dump_json(indent=2))
            Path(tmp_name).replace(self._manifest_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _upsert(self, record: SchemaRecord) -> None:
        manifest = self._load_manifest()
        for i, existing in enumerate(manifest.schemas):
            if existing.name == record.name:
                manifest.schemas[i] = record
                break
        else:
            manifest.schemas.append(record)
        self._save_manifest(manifest)

    def write_draft(self, name: str, json_schema: JsonSchemaValue) -> SchemaRecord:
        """Generate a Pydantic model from a JSON Schema and store it as a `draft` pending validation.

        The schema is converted into a module by `datamodel-code-generator`, written as
        `<name>.v<version>.py`, and imported back to check that it yields a usable model.

        Args:
            name: Schema name matching `^[a-z][a-z0-9_]*$`. Reusing a name creates a new version.
            json_schema: The JSON Schema document to convert.

        Returns:
            The manifest record for the new draft version.

        Raises:
            ValueError: If `name` is not a valid schema name.
            ModelGenerationError: If the schema does not convert into an importable model.
        """
        if not _NAME_RE.match(name):
            raise ValueError(f'Invalid schema name {name!r}: must match {_NAME_RE.pattern!r}.')
        self.directory.mkdir(parents=True, exist_ok=True)
        now = _utcnow()
        existing = self.get(name)
        version = existing.version + 1 if existing is not None else 1
        class_name = model_class_name(name)
        source = generate_model_source(json_schema, class_name=class_name, input_filename=f'{name}.v{version}.json')
        source_file = f'{name}.v{version}.py'
        path = self.directory / source_file
        path.write_text(source, encoding='utf-8')
        try:
            load_model(path, class_name)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        record = SchemaRecord(
            name=name,
            source_file=source_file,
            class_name=class_name,
            version=version,
            status='draft',
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            activated_at=existing.activated_at if existing is not None else None,
        )
        self._upsert(record)
        return record

    def get(self, name: str) -> SchemaRecord | None:
        """Return the manifest record for `name`, or `None` if it was never authored."""
        for record in self._load_manifest().schemas:
            if record.name == name:
                return record
        return None

    def read_model(self, record: SchemaRecord) -> type[BaseModel]:
        """Import the generated module for a manifest record and return its model class."""
        return load_model(self.directory / record.source_file, record.class_name)

    def read_source(self, record: SchemaRecord) -> str:
        """Read the source code of the generated model module for a manifest record."""
        return (self.directory / record.source_file).read_text(encoding='utf-8')

    def set_error(self, name: str, error: str | None) -> None:
        """Record (or clear) the last lint/evaluation error for a schema."""
        manifest = self._load_manifest()
        changed = False
        for record in manifest.schemas:
            if record.name == name:
                record.last_error = error
                changed = True
        if changed:
            self._save_manifest(manifest)

    def activate(self, name: str) -> SchemaRecord:
        """Mark `name` as the single active schema, demoting any other active schema to `inactive`.

        Raises:
            ValueError: If no schema named `name` exists.
        """
        manifest = self._load_manifest()
        target: SchemaRecord | None = None
        now = _utcnow()
        for record in manifest.schemas:
            if record.name == name:
                target = record
            elif record.status == 'active':
                record.status = 'inactive'
        if target is None:
            raise ValueError(f'No schema named {name!r} exists.')
        target.status = 'active'
        target.activated_at = now
        target.updated_at = now
        target.last_error = None
        self._save_manifest(manifest)
        return target

    def deactivate(self, name: str) -> bool:
        """Mark `name` as `inactive`. Returns whether the schema existed."""
        manifest = self._load_manifest()
        for record in manifest.schemas:
            if record.name == name:
                record.status = 'inactive'
                record.updated_at = _utcnow()
                self._save_manifest(manifest)
                return True
        return False

    def list_all(self) -> list[SchemaRecord]:
        """Return every manifest record, in insertion order."""
        return self._load_manifest().schemas

    def active_record(self) -> SchemaRecord | None:
        """Return the currently active schema record, or `None`."""
        for record in self._load_manifest().schemas:
            if record.status == 'active':
                return record
        return None

    def active_output_type(self) -> type[BaseModel] | None:
        """Return the generated Pydantic model of the active schema, ready to use as an output type.

        Use this to thread the evolved output type into the next `agent.run(...)`, or into a
        separate consumer agent. Returns `None` when no schema is active.
        """
        record = self.active_record()
        return self.read_model(record) if record is not None else None
