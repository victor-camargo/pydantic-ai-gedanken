# pyright: reportPrivateUsage=false
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_gedanken.capabilities.dynamic_output import (
    AUTHOR_TOOL_NAME,
    DynamicOutput,
    DynamicOutputToolset,
    ModelGenerationError,
    SchemaStore,
    VerificationResult,
    Verifier,
    generate_model_source,
    lint_json_schema,
    load_model,
    model_class_name,
    run_verifiers,
)

PERSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'title': 'Person',
    'description': 'A person.',
    'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
    'required': ['name', 'age'],
}

PERSON_SCHEMA_V2: dict[str, Any] = {
    **PERSON_SCHEMA,
    'properties': {**PERSON_SCHEMA['properties'], 'email': {'type': 'string'}},
}

NESTED_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'title': 'Task',
    'description': 'A task.',
    'properties': {
        'level': {'type': 'string', 'enum': ['low', 'high']},
        'owner': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
    },
    'required': ['level', 'owner'],
}


def make_verifier(required_field: str) -> Verifier:
    async def check(model: type[BaseModel]) -> VerificationResult:
        ok = required_field in model.model_fields
        message = 'ok' if ok else f'model must declare a {required_field!r} field'
        return VerificationResult(passed=ok, message=message)

    return Verifier(name=f'requires_{required_field}', check_fn=check)


def counting_verifier() -> tuple[list[type[BaseModel]], Verifier]:
    """A passing verifier that records the models it was handed."""
    calls: list[type[BaseModel]] = []

    async def check(model: type[BaseModel]) -> VerificationResult:
        calls.append(model)
        return VerificationResult(passed=True, message='ok')

    return calls, Verifier(name='counted', check_fn=check)


def person_model() -> type[BaseModel]:
    class Person(BaseModel):
        name: str
        age: int

    return Person


def fake_ctx(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {'agent': None, 'messages': [], 'deps': None}
    return SimpleNamespace(**{**defaults, **kwargs})


class FakeResult:
    def __init__(self, output: Any) -> None:
        self.output = output

    def all_messages(self) -> list[ModelMessage]:
        return []


class FakeAgent:
    def __init__(self, on_run: Any = None) -> None:
        self.prompts: list[str] = []
        self._on_run = on_run

    async def run(self, prompt: str, message_history: Any = None, deps: Any = None) -> FakeResult:
        self.prompts.append(prompt)
        if self._on_run is not None:
            self._on_run()
        return FakeResult(f'retry-{len(self.prompts)}')


# --- Code generation ---


def test_model_class_name():
    assert model_class_name('person') == 'Person'
    assert model_class_name('avaliacao_tarefas') == 'AvaliacaoTarefas'


def test_generate_model_source_renders_a_pydantic_model():
    source = generate_model_source(PERSON_SCHEMA, class_name='Person', input_filename='person.v1.json')
    assert 'class Person(BaseModel):' in source
    assert 'name: str' in source
    assert source.endswith('\n')


def test_generate_model_source_rejects_unusable_schema():
    with pytest.raises(ModelGenerationError, match='Could not generate'):
        generate_model_source({'$ref': '#/$defs/missing'}, class_name='Person', input_filename='p.json')


def test_load_model_caches_until_the_file_changes(tmp_path: Path):
    store = SchemaStore(tmp_path)
    record = store.write_draft('person', PERSON_SCHEMA)
    path = tmp_path / record.source_file
    first = load_model(path, 'Person')
    assert load_model(path, 'Person') is first
    path.write_text(store.read_source(record).replace('age: int', 'age: int\n    email: str | None = None'))
    reloaded = load_model(path, 'Person')
    assert reloaded is not first
    assert 'email' in reloaded.model_fields


def test_load_model_resolves_forward_references(tmp_path: Path):
    store = SchemaStore(tmp_path)
    record = store.write_draft('task', NESTED_SCHEMA)
    model = store.read_model(record)
    schema = model.model_json_schema(by_alias=False)
    assert set(schema['properties']) == {'level', 'owner'}
    assert set(schema['$defs']) == {'Level', 'Owner'}
    instance = model(level='high', owner={'name': 'Ada'})
    assert instance.model_dump() == {'level': 'high', 'owner': {'name': 'Ada'}}


def test_load_model_rejects_unresolvable_annotations(tmp_path: Path):
    path = tmp_path / 'dangling.v1.py'
    path.write_text(
        'from __future__ import annotations\n'
        'from pydantic import BaseModel\n'
        'class Dangling(BaseModel):\n'
        '    other: Missing\n'
    )
    with pytest.raises(ModelGenerationError, match='not fully defined'):
        load_model(path, 'Dangling')


def test_load_model_keeps_the_previous_module_on_a_bad_edit(tmp_path: Path):
    store = SchemaStore(tmp_path)
    record = store.write_draft('task', NESTED_SCHEMA)
    path = tmp_path / record.source_file
    loaded = load_model(path, 'Task')
    path.write_text('class Task:\n    pass\n')
    with pytest.raises(ModelGenerationError, match='defines no Pydantic model'):
        load_model(path, 'Task')
    assert loaded.model_json_schema(by_alias=False)['properties'].keys() == {'level', 'owner'}


def test_load_model_rejects_broken_module(tmp_path: Path):
    path = tmp_path / 'broken.v1.py'
    path.write_text('this is not python')
    with pytest.raises(ModelGenerationError, match='could not be imported'):
        load_model(path, 'Broken')


def test_load_model_rejects_module_without_the_model(tmp_path: Path):
    path = tmp_path / 'empty.v1.py'
    path.write_text('x = 1\n')
    with pytest.raises(ModelGenerationError, match='defines no Pydantic model'):
        load_model(path, 'Empty')


# --- SchemaStore ---


def test_store_write_draft_versioning_and_timestamps(tmp_path: Path):
    store = SchemaStore(tmp_path)
    first = store.write_draft('person', PERSON_SCHEMA)
    assert (first.version, first.status) == (1, 'draft')
    assert (first.source_file, first.class_name) == ('person.v1.py', 'Person')
    assert first.created_at == first.updated_at
    second = store.write_draft('person', PERSON_SCHEMA)
    assert second.version == 2
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert (tmp_path / 'person.v1.py').exists()
    assert (tmp_path / 'person.v2.py').exists()
    assert set(store.read_model(second).model_fields) == {'name', 'age'}
    assert 'class Person(BaseModel):' in store.read_source(second)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert [r['name'] for r in manifest['schemas']] == ['person']


def test_store_invalid_name(tmp_path: Path):
    store = SchemaStore(tmp_path)
    with pytest.raises(ValueError, match='Invalid schema name'):
        store.write_draft('Not-Valid', PERSON_SCHEMA)


def test_store_write_draft_discards_unusable_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = SchemaStore(tmp_path)

    def broken_source(*args: Any, **kwargs: Any) -> str:
        return 'x = 1\n'

    monkeypatch.setattr('pydantic_ai_gedanken.capabilities.dynamic_output._store.generate_model_source', broken_source)
    with pytest.raises(ModelGenerationError, match='defines no Pydantic model'):
        store.write_draft('person', PERSON_SCHEMA)
    assert not (tmp_path / 'person.v1.py').exists()
    assert store.get('person') is None


def test_store_activate_demotes_previous_and_deactivate(tmp_path: Path):
    store = SchemaStore(tmp_path)
    store.write_draft('first', PERSON_SCHEMA)
    store.write_draft('second', PERSON_SCHEMA)
    activated = store.activate('first')
    assert activated.status == 'active'
    assert activated.activated_at is not None
    store.activate('second')
    statuses = {r.name: r.status for r in store.list_all()}
    assert statuses == {'first': 'inactive', 'second': 'active'}
    record = store.active_record()
    assert record is not None and record.name == 'second'
    output_type = store.active_output_type()
    assert output_type is not None and output_type.__name__ == 'Second'
    assert store.deactivate('second') is True
    assert store.active_record() is None
    assert store.active_output_type() is None
    assert store.deactivate('missing') is False
    with pytest.raises(ValueError, match='No schema named'):
        store.activate('missing')


def test_store_set_error_and_corrupt_manifest(tmp_path: Path):
    store = SchemaStore(tmp_path)
    store.write_draft('person', PERSON_SCHEMA)
    store.set_error('person', 'boom')
    record = store.get('person')
    assert record is not None and record.last_error == 'boom'
    store.set_error('person', None)
    record = store.get('person')
    assert record is not None and record.last_error is None
    store.set_error('ghost', 'nowhere to put this')
    record = store.get('person')
    assert record is not None and record.last_error is None
    (tmp_path / 'manifest.json').write_text('{not json')
    assert store.list_all() == []


def test_store_manifest_write_leaves_no_temporary_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = SchemaStore(tmp_path)

    def failing_replace(self: Path, target: Any) -> Path:
        raise OSError('no space left on device')

    monkeypatch.setattr(Path, 'replace', failing_replace)
    with pytest.raises(OSError, match='no space left on device'):
        store.write_draft('person', PERSON_SCHEMA)
    assert not list(tmp_path.glob('manifest.*.json.tmp'))


# --- Lint ---


def test_lint_valid_schema():
    result = lint_json_schema(PERSON_SCHEMA)
    assert result.ok and not result.warnings


def test_lint_rejects_non_object_root():
    result = lint_json_schema({'type': 'array'})
    assert not result.ok


def test_lint_rejects_undeclared_required():
    result = lint_json_schema({'type': 'object', 'properties': {}, 'required': ['ghost']})
    assert any('ghost' in error for error in result.errors)


def test_lint_rejects_meta_schema_violation():
    result = lint_json_schema({'type': 'object', 'properties': {'a': {'type': 'nope'}}})
    assert any('Meta-schema violation' in error for error in result.errors)


def test_lint_rejects_non_dict_property():
    result = lint_json_schema({'type': 'object', 'properties': {'a': 'nope'}})
    assert not result.ok
    boolean_property = lint_json_schema({'type': 'object', 'properties': {'a': True}})
    assert any('must be a schema object' in error for error in boolean_property.errors)


def test_lint_warns_without_properties():
    result = lint_json_schema({'type': 'object', 'title': 'Anything', 'description': 'Anything at all.'})
    assert result.ok
    assert any('no "properties"' in w for w in result.warnings)


def test_lint_warnings_for_style_issues():
    result = lint_json_schema({'type': 'object', 'properties': {'a': {}}})
    assert result.ok
    assert any('title' in w for w in result.warnings)
    assert any('description' in w for w in result.warnings)
    assert any('type constraint' in w for w in result.warnings)


# --- Verifiers ---


def test_run_verifiers_parallel_and_sequential():
    verifiers = [make_verifier('name'), make_verifier('email')]
    model = person_model()
    parallel = asyncio.run(run_verifiers(verifiers, model, parallel=True))
    sequential = asyncio.run(run_verifiers(verifiers, model, parallel=False))
    expected = [('requires_email', "model must declare a 'email' field")]
    assert parallel == expected
    assert sequential == expected
    assert asyncio.run(run_verifiers([make_verifier('name')], model)) == []


# --- Authoring toolset ---


def test_author_tool_rejects_bad_name(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    with pytest.raises(ModelRetry, match='Invalid schema name'):
        asyncio.run(toolset.author_output_schema('Bad Name', PERSON_SCHEMA))


def test_author_tool_rejects_lint_failure(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    with pytest.raises(ModelRetry, match='format/lint'):
        asyncio.run(toolset.author_output_schema('person', {'type': 'array'}))
    assert capability.store.get('person') is None


def test_author_tool_rejects_ungeneratable_schema(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    schema = {'type': 'object', 'properties': {'a': {'$ref': '#/$defs/missing'}}}
    with pytest.raises(ModelRetry, match='Could not generate'):
        asyncio.run(toolset.author_output_schema('person', schema))
    assert capability.store.get('person') is None


def test_author_tool_activates_immediately_and_returns_the_model(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    message = asyncio.run(toolset.author_output_schema('person', PERSON_SCHEMA))
    assert 'active output type' in message
    assert 'class Person(BaseModel):' in message
    assert capability._schema_created is True
    record = capability.store.active_record()
    assert record is not None and (record.name, record.version, record.status) == ('person', 1, 'active')
    assert toolset.read_source(record) in message


def test_author_tool_includes_lint_warnings(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    message = asyncio.run(toolset.author_output_schema('person', {'type': 'object', 'properties': {'a': {}}}))
    assert 'Lint warnings' in message


def test_list_and_deactivate_tools(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    toolset = DynamicOutputToolset(capability.store, capability)
    assert 'No output schemas' in asyncio.run(toolset.list_output_schemas())
    capability.store.write_draft('person', PERSON_SCHEMA)
    listing = asyncio.run(toolset.list_output_schemas())
    assert '"person"' in listing
    assert 'now inactive' in asyncio.run(toolset.deactivate_output_schema('person'))
    assert 'No output schema named' in asyncio.run(toolset.deactivate_output_schema('ghost'))


# --- Capability wiring ---


def test_capability_opts_out_of_spec_construction():
    assert DynamicOutput.get_serialization_name() is None


# --- prepare_output_tools ---


def test_prepare_output_tools_erases_until_activation(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    capability._schema_created = True
    tool_defs = [ToolDefinition(name='final_result', kind='output')]
    result = asyncio.run(capability.prepare_output_tools(fake_ctx(), tool_defs))
    assert result == []
    assert capability._schema_created is False

    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')
    served = asyncio.run(capability.prepare_output_tools(fake_ctx(), tool_defs))
    assert len(served) == 1
    schema = served[0].parameters_json_schema
    assert schema['type'] == 'object'
    assert set(schema['properties']) == {'name', 'age'}
    assert served[0].description == 'A person.'


def test_prepare_output_tools_keeps_description_without_a_docstring(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    capability.store.write_draft('person', {**PERSON_SCHEMA, 'description': None})
    capability.store.activate('person')
    tool_defs = [ToolDefinition(name='final_result', kind='output', description='fallback')]
    served = asyncio.run(capability.prepare_output_tools(fake_ctx(), tool_defs))
    assert served[0].description == 'fallback'


# --- wrap_run verification loop ---


def test_wrap_run_without_verifiers_passes_through(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path)
    original = FakeResult('done')

    async def handler() -> Any:
        return original

    assert asyncio.run(capability.wrap_run(fake_ctx(), handler=handler)) is original


def test_wrap_run_without_active_schema_skips_verification(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path, verifiers=[make_verifier('email')])
    original = FakeResult('done')

    async def handler() -> Any:
        return original

    assert asyncio.run(capability.wrap_run(fake_ctx(agent=FakeAgent()), handler=handler)) is original


def test_wrap_run_verification_passes(tmp_path: Path):
    calls, verifier = counting_verifier()
    capability = DynamicOutput(directory=tmp_path, verifiers=[verifier])
    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')
    capability.store.set_error('person', 'stale')
    original = FakeResult('done')

    async def handler() -> Any:
        return original

    agent = FakeAgent()
    result = asyncio.run(capability.wrap_run(fake_ctx(agent=agent), handler=handler))
    assert result is original
    assert agent.prompts == []
    # The verifier is handed the generated Pydantic model, not the authored JSON Schema.
    assert calls == [capability.store.active_output_type()]
    record = capability.store.get('person')
    assert record is not None and record.last_error is None


def test_wrap_run_retries_with_feedback_until_pass(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path, verifiers=[make_verifier('email')])
    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')

    def evolve_schema() -> None:
        capability.store.write_draft('person', PERSON_SCHEMA_V2)
        capability.store.activate('person')

    agent = FakeAgent(on_run=evolve_schema)

    async def handler() -> Any:
        return FakeResult('first attempt')

    result = asyncio.run(capability.wrap_run(fake_ctx(agent=agent), handler=handler))
    assert result.output == 'retry-1'
    assert len(agent.prompts) == 1
    assert 'requires_email' in agent.prompts[0]
    assert AUTHOR_TOOL_NAME in agent.prompts[0]
    assert capability._in_retry is False
    record = capability.store.get('person')
    assert record is not None and record.version == 2 and record.last_error is None


def test_wrap_run_warns_when_retries_exhausted(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path, verifiers=[make_verifier('email')], max_retries=2)
    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')
    agent = FakeAgent()

    async def handler() -> Any:
        return FakeResult('first attempt')

    with pytest.warns(UserWarning, match='still failing after 2 retries'):
        result = asyncio.run(capability.wrap_run(fake_ctx(agent=agent), handler=handler))
    assert result.output == 'retry-2'
    assert len(agent.prompts) == 2
    record = capability.store.get('person')
    assert record is not None and record.last_error is not None


def test_wrap_run_retry_run_passes_through(tmp_path: Path):
    calls, verifier = counting_verifier()
    capability = DynamicOutput(directory=tmp_path, verifiers=[verifier])
    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')
    capability._in_retry = True
    original = FakeResult('done')

    async def handler() -> Any:
        return original

    assert asyncio.run(capability.wrap_run(fake_ctx(agent=FakeAgent()), handler=handler)) is original
    assert calls == []


def test_wrap_run_accepts_the_model_evolved_on_the_last_retry(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path, verifiers=[make_verifier('email')], max_retries=1)
    capability.store.write_draft('person', PERSON_SCHEMA)
    capability.store.activate('person')

    def evolve_schema() -> None:
        capability.store.write_draft('person', PERSON_SCHEMA_V2)
        capability.store.activate('person')

    agent = FakeAgent(on_run=evolve_schema)

    async def handler() -> Any:
        return FakeResult('first attempt')

    result = asyncio.run(capability.wrap_run(fake_ctx(agent=agent), handler=handler))
    assert result.output == 'retry-1'
    record = capability.store.get('person')
    assert record is not None and record.version == 2 and record.last_error is None


# --- End-to-end with a scripted model ---


def test_agent_evolves_output_type_end_to_end(tmp_path: Path):
    capability = DynamicOutput(directory=tmp_path, verifiers=[make_verifier('name'), make_verifier('email')])

    def scripted_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        returned = {part.tool_name for part in last.parts if isinstance(part, ToolReturnPart)}
        if AUTHOR_TOOL_NAME in returned:
            schema = {tool.name: tool for tool in info.output_tools}['final_result'].parameters_json_schema
            args: dict[str, Any] = {'name': 'Ada', 'age': 36}
            if 'email' in schema.get('properties', {}):
                args['email'] = 'ada@example.com'
            return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=args)])
        prompts = ' '.join(str(part.content) for part in last.parts if isinstance(part, UserPromptPart))
        schema = PERSON_SCHEMA_V2 if 'verification failed' in prompts else PERSON_SCHEMA
        return ModelResponse(
            parts=[ToolCallPart(tool_name=AUTHOR_TOOL_NAME, args={'name': 'person', 'json_schema': schema})]
        )

    agent = Agent(
        FunctionModel(scripted_llm),
        capabilities=[capability],
        output_type=[str, DynamicOutput.placeholder_output_type()],
    )
    result = agent.run_sync('Evolve an output schema for a person and produce Ada.')
    assert result.output == {'name': 'Ada', 'age': 36, 'email': 'ada@example.com'}
    record = capability.store.active_record()
    assert record is not None and (record.name, record.version, record.status) == ('person', 2, 'active')
    assert record.last_error is None
    output_type = capability.store.active_output_type()
    assert output_type is not None
    assert output_type(**result.output).model_dump() == result.output
