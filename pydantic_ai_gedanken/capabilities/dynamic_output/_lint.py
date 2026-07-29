"""Format and lint checks for agent-authored JSON Schemas.

A schema must pass these checks before it is converted into a Pydantic model and activated
as an output type:

- structural format: the root must be a JSON object schema (`"type": "object"`);
- meta-schema conformance: the document must be a valid JSON Schema per its declared
  (or the 2020-12 default) meta-schema;
- lint rules: `required` entries must name declared properties, and stylistic issues
  (missing `title`/`description`, untyped properties) are reported as warnings.

Whether the schema actually converts into an importable model is decided by the code
generator itself, in [`_codegen`][pydantic_ai_gedanken.capabilities.dynamic_output], when the
draft is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for
from pydantic.json_schema import JsonSchemaValue


@dataclass
class SchemaLintResult:
    """Outcome of linting a JSON Schema: blocking errors and non-blocking warnings."""

    errors: list[str] = field(default_factory=list[str])
    """Problems that make the schema unusable as an output type."""
    warnings: list[str] = field(default_factory=list[str])
    """Stylistic or robustness issues that do not block activation."""

    @property
    def ok(self) -> bool:
        """Whether the schema is free of blocking errors."""
        return not self.errors


def _check_meta_schema(schema: JsonSchemaValue, result: SchemaLintResult) -> None:
    validator_cls = validator_for(schema, default=Draft202012Validator)
    meta_validator = validator_cls(validator_cls.META_SCHEMA)
    for error in meta_validator.iter_errors(schema):
        path = '/'.join(str(part) for part in error.absolute_path) or '<root>'
        result.errors.append(f'Meta-schema violation at {path}: {error.message}')


def _check_properties(schema: JsonSchemaValue, result: SchemaLintResult) -> None:
    raw_properties = schema.get('properties')
    if not isinstance(raw_properties, dict):
        result.warnings.append('Root schema declares no "properties"; the output type will accept any object.')
    properties = cast('dict[str, object]', raw_properties) if isinstance(raw_properties, dict) else {}
    raw_required = schema.get('required')
    if isinstance(raw_required, list):
        undeclared = [str(name) for name in cast('list[object]', raw_required) if name not in properties]
        if undeclared:
            result.errors.append(f'"required" lists properties that are not declared: {", ".join(sorted(undeclared))}.')
    typed_keys = ('type', '$ref', 'enum', 'const', 'anyOf', 'oneOf', 'allOf')
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            result.errors.append(f'Property {prop_name!r} must be a schema object, got {type(prop_schema).__name__}.')
        elif not any(key in prop_schema for key in typed_keys):
            result.warnings.append(
                f'Property {prop_name!r} has no type constraint (missing "type", "$ref", "enum", …).'
            )
    if 'title' not in schema:
        result.warnings.append('Root schema has no "title"; a title names the output type for the model.')
    if 'description' not in schema:
        result.warnings.append('Root schema has no "description"; a description helps the model fill the output.')


def lint_json_schema(schema: JsonSchemaValue) -> SchemaLintResult:
    """Run all format and lint checks on a JSON Schema document.

    Args:
        schema: The JSON Schema to check.

    Returns:
        A [`SchemaLintResult`][pydantic_ai_gedanken.capabilities.dynamic_output.SchemaLintResult]
        with blocking `errors` and non-blocking `warnings`.
    """
    result = SchemaLintResult()
    if schema.get('type') != 'object':
        result.errors.append('The root of an output schema must be an object schema (`"type": "object"`).')
        return result
    _check_meta_schema(schema, result)
    if result.errors:
        return result
    _check_properties(schema, result)
    return result
