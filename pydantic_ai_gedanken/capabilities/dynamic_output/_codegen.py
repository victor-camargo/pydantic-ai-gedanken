"""Conversion of agent-authored JSON Schemas into Pydantic model modules.

The agent authors a JSON Schema, but the artifact that is stored and served as the output
type is a Pydantic model: [`datamodel-code-generator`](https://koxudaxi.github.io/datamodel-code-generator/)
turns the schema into a plain Python module (`<name>.v<version>.py`) which is then imported
back to recover the model class. Working with a generated model rather than a raw JSON
document keeps the output type readable for the model and hand-editable on disk.
"""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Final

from datamodel_code_generator import InputFileType, generate
from datamodel_code_generator.enums import DataModelType
from datamodel_code_generator.format import Formatter
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

MODEL_PRESET: Final = 'standard-py312-20260619'
"""`datamodel-code-generator` preset pinning the style of the generated modules."""

_MODULE_PREFIX: Final = 'pydantic_ai_gedanken_dynamic_output.'

_MODEL_CACHE: dict[tuple[str, int, int], type[BaseModel]] = {}
"""Loaded model classes keyed by `(path, mtime, size)`, so edits on disk are picked up."""


class ModelGenerationError(Exception):
    """Raised when a JSON Schema cannot be turned into a usable Pydantic model."""


def model_class_name(name: str) -> str:
    """Name of the root model class generated for a schema name (`avaliacao_tarefas` → `AvaliacaoTarefas`)."""
    return ''.join(part[:1].upper() + part[1:] for part in name.split('_'))


def generate_model_source(json_schema: JsonSchemaValue, *, class_name: str, input_filename: str) -> str:
    """Convert a JSON Schema into the source code of a Pydantic model module.

    Args:
        json_schema: The JSON Schema document to convert.
        class_name: Name to give the generated root model class.
        input_filename: Name recorded in the generated file header, for provenance.

    Returns:
        The source code of a module defining `class_name` (plus any nested models and enums).

    Raises:
        ModelGenerationError: If the schema cannot be converted.
    """
    try:
        source = generate(
            json.dumps(json_schema),
            input_file_type=InputFileType.JsonSchema,
            input_filename=input_filename,
            output_model_type=DataModelType.PydanticV2BaseModel,
            preset=MODEL_PRESET,
            class_name=class_name,
            use_schema_description=True,
            use_field_description=True,
            use_attribute_docstrings=True,
            snake_case_field=True,
            # A dangling `$ref` would otherwise silently degrade a field to `Any`; failing here
            # sends the schema back to the model instead.
            strict_refs=True,
            formatters=[Formatter.BUILTIN],
        )
    except Exception as exc:
        raise ModelGenerationError(f'Could not generate a Pydantic model from the JSON Schema: {exc}') from exc
    if not isinstance(source, str):  # pragma: no cover — a single JSON Schema always yields a single module
        raise ModelGenerationError('Model generation produced multiple modules; expected a single one.')
    return source if source.endswith('\n') else source + '\n'


def _module_name(path: Path) -> str:
    """Stable, collision-free module name for a generated file: same path → same name."""
    digest = sha256(str(path.resolve()).encode()).hexdigest()[:8]
    return f'{_MODULE_PREFIX}{path.stem.replace(".", "_")}_{digest}'


def _exec_module(module: ModuleType, path: Path, class_name: str) -> type[BaseModel]:
    """Execute a generated module and pull the `class_name` model out of it."""
    try:
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    except Exception as exc:
        raise ModelGenerationError(f'Generated module {path.name} could not be imported: {exc}') from exc
    model: object = getattr(module, class_name, None)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ModelGenerationError(f'Generated module {path.name} defines no Pydantic model named {class_name!r}.')
    try:
        # Resolve any forward reference now, so a model that cannot be built fails here
        # rather than when the output tool is served.
        model.model_rebuild()
    except Exception as exc:
        raise ModelGenerationError(f'Model {class_name!r} in {path.name} is not fully defined: {exc}') from exc
    return model


def load_model(path: Path, class_name: str) -> type[BaseModel]:
    """Import a generated module and return the model class it defines.

    The module is executed under its own name in `sys.modules` — which is what lets Pydantic
    resolve the forward references the generated code relies on — and no bytecode is written
    next to it. The result is cached on the file's modification time and size, so a module
    edited by hand is re-imported on the next call.

    Args:
        path: Path to the generated module.
        class_name: Name of the root model class to pull out of it.

    Raises:
        ModelGenerationError: If the module cannot be imported or defines no usable model.
    """
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    module = ModuleType(_module_name(path))
    module.__file__ = str(path)
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        model = _exec_module(module, path, class_name)
    except ModelGenerationError:
        # Leave the last module that did load in place: a bad hand-edit must not break the
        # model already handed out for this file.
        if previous is None:
            del sys.modules[module.__name__]
        else:
            sys.modules[module.__name__] = previous
        raise
    _MODEL_CACHE[key] = model
    return model
