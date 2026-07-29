"""Verification of agent-authored output models via user-supplied check functions.

Each schema evolution is checked by a list of
[`Verifier`][pydantic_ai_gedanken.capabilities.dynamic_output.Verifier]s: named async
functions that receive the generated Pydantic model and return a
[`VerificationResult`][pydantic_ai_gedanken.capabilities.dynamic_output.VerificationResult].
Failures are fed back to the model so the output type can evolve.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class VerificationResult:
    """The outcome of a single schema verification check."""

    passed: bool
    """Whether the check passed."""
    message: str
    """A human-readable description of the outcome."""


@dataclass
class Verifier:
    """A named verification check to run against a candidate output model."""

    name: str
    """A short identifier for this verifier (e.g. `'required_fields'`, `'naming'`)."""
    check_fn: Callable[[type[BaseModel]], Awaitable[VerificationResult]]
    """An async callable that receives the generated Pydantic model (inspect it via
    `model.model_fields` or `model.model_json_schema()`) and returns a
    [`VerificationResult`][pydantic_ai_gedanken.capabilities.dynamic_output.VerificationResult]."""


async def run_verifiers(
    verifiers: Sequence[Verifier],
    model: type[BaseModel],
    *,
    parallel: bool = True,
) -> list[tuple[str, str]]:
    """Run all verifiers against a candidate model and return `(name, message)` for each failure.

    Args:
        verifiers: The verification checks to run.
        model: The generated Pydantic model passed to each `check_fn`.
        parallel: When `True` (the default), verifiers execute concurrently via
            `asyncio.gather`; when `False`, they run sequentially in list order.
    """
    if parallel and len(verifiers) > 1:
        results = await asyncio.gather(*(verifier.check_fn(model) for verifier in verifiers))
        return [
            (verifier.name, result.message)
            for verifier, result in zip(verifiers, results, strict=True)
            if not result.passed
        ]
    failures: list[tuple[str, str]] = []
    for verifier in verifiers:
        result = await verifier.check_fn(model)
        if not result.passed:
            failures.append((verifier.name, result.message))
    return failures
