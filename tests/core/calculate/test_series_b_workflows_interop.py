"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import asyncio
import gc
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ase.calculators.mixing import SumCalculator

from fairchem.core.calculate.series_b_workflows import (
    CoarseRefineOptions,
    DyNEBOptions,
    RESPAOptions,
    run_coarse_refine,
    run_dyneb,
    run_respa,
)
from tests.core.calculate.test_series_b_workflows import Harmonic, _atom, _band


@pytest.mark.parametrize("array_name", ["positions", "momenta"])
def test_readonly_inplace_arrays_are_rejected_before_work_or_partial_update(array_name):
    atoms = _atom()
    before = atoms.copy()
    atoms.arrays[array_name].flags.writeable = False
    calls = []

    def factory():
        calls.append(True)
        return Harmonic()

    with pytest.raises(ValueError):
        run_respa(
            atoms,
            teacher_factory=factory,
            student_factory=factory,
            options=RESPAOptions(0.05),
            outer_steps=1,
            inplace=True,
        )
    np.testing.assert_array_equal(atoms.positions, before.positions)
    np.testing.assert_array_equal(atoms.get_momenta(), before.get_momenta())
    assert not calls


class FailingHarmonic(Harmonic):
    def __init__(self, error_type):
        super().__init__()
        self.error_type = error_type
        # Model wrappers commonly own bound callables and therefore cycles.
        self.bound = self.calculate

    def calculate(self, *args, **kwargs):
        try:
            self.fail()
        except ValueError as error:
            if self.error_type is ExceptionGroup:
                errors = [error]
            else:
                raise self.error_type("job interrupted") from error
        # Raise outside the except block so the leaf error is reachable only
        # through the group's children, not through __cause__ or __context__.
        raise ExceptionGroup("job interrupted", errors)

    def fail(self):
        raise ValueError("inner calculation failure")


@pytest.mark.parametrize(
    "workflow", ["respa", "coarse", "reference", "dyneb", "dyneb_reference"]
)
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, ExceptionGroup]
)
def test_job_exception_records_do_not_keep_private_calculators_alive(
    workflow, error_type
):
    atoms = _atom()
    original = atoms.copy()
    existing = Harmonic()
    atoms.calc = existing
    references = []
    errors = []

    def factory(fail=False):
        calculator = FailingHarmonic(error_type) if fail else Harmonic()
        references.append(weakref.ref(calculator))
        return calculator

    try:
        if workflow == "respa":
            run_respa(
                atoms,
                teacher_factory=lambda: factory(True),
                student_factory=factory,
                options=RESPAOptions(0.02),
                outer_steps=1,
                inplace=True,
            )
        elif workflow in {"coarse", "reference"}:
            run_coarse_refine(
                atoms,
                coarse_factory=lambda: factory(workflow == "coarse"),
                reference_factory=lambda: factory(workflow == "reference"),
                options=CoarseRefineOptions(coarse_steps=0, reference_steps=0),
            )
        else:
            images = _band()
            run_dyneb(
                images,
                calculator_factory=lambda: factory(workflow == "dyneb"),
                reference_calculator_factory=lambda: factory(
                    workflow == "dyneb_reference"
                ),
                options=DyNEBOptions(max_steps=0, final_refine=False),
            )
    except error_type as error:
        errors.append(error)
    assert len(errors) == 1
    assert str(errors[0]).startswith("job interrupted")
    if error_type is ExceptionGroup:
        assert isinstance(errors[0].exceptions[0], ValueError)
    else:
        assert isinstance(errors[0].__cause__, ValueError)
    assert any(
        frame.name == "calculate"
        for frame in traceback.extract_tb(errors[0].__traceback__)
    )
    gc.collect()
    assert references
    assert all(reference() is None for reference in references)
    assert atoms.calc is existing
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.get_momenta(), original.get_momenta())


def test_successful_respa_releases_cyclic_models_before_the_next_job():
    references = []

    def factory():
        calculator = Harmonic()
        calculator.bound = calculator.calculate
        references.append(weakref.ref(calculator))
        return calculator

    # The workflow must release the private models itself, including when the
    # caller manages collection explicitly to reduce latency in long MD jobs.
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = run_respa(
            _atom(),
            teacher_factory=factory,
            student_factory=factory,
            options=RESPAOptions(0.02),
            outer_steps=1,
        )
        assert references
        assert all(reference() is None for reference in references)
        assert result.atoms.calc is None
        np.testing.assert_allclose(result.reference_forces, -result.atoms.positions)
    finally:
        if enabled:
            gc.enable()


def test_concurrent_mixed_calculator_jobs_preserve_results_and_source_metadata():
    atoms = _atom()
    original = atoms.copy()

    def job(center):
        def factory(stiffness):
            return SumCalculator(
                [Harmonic(stiffness, center), Harmonic(stiffness, center)]
            )

        return run_coarse_refine(
            atoms,
            coarse_factory=lambda: factory(0.1),
            reference_factory=lambda: factory(1.5),
            options=CoarseRefineOptions(coarse_steps=1, reference_steps=4),
        )

    centers = (-0.3, 0.4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(job, centers))
    for result, center in zip(results, centers):
        displacement = result.atoms.positions - np.array([center, 0.0, 0.0])
        np.testing.assert_allclose(result.reference_forces, -3.0 * displacement)
        assert result.reference_energy == pytest.approx(
            1.5 * np.square(displacement).sum()
        )
        # Result arrays must not alias the live calculator's ASE property cache.
        result.reference_forces[:] = 123
        np.testing.assert_allclose(result.atoms.get_forces(), -3.0 * displacement)
    results[0].atoms.info["metadata"]["array"][0] = 9
    assert results[1].atoms.info["metadata"]["array"][0] == 1
    assert atoms.info["metadata"]["array"][0] == 1
    assert results[0].atoms.calc is not results[1].atoms.calc
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.get_momenta(), original.get_momenta())


def test_chained_error_cleanup_does_not_close_a_suspended_caller_generator():
    closed = []

    def source():
        try:
            try:
                raise ValueError("previous unrelated error")
            except ValueError as error:
                yield error
            yield "still usable"
        finally:
            closed.append(True)

    generator = source()
    previous = next(generator)

    def factory():
        raise RuntimeError("factory failed") from previous

    try:
        with pytest.raises(RuntimeError, match="factory failed"):
            run_respa(
                _atom(),
                teacher_factory=factory,
                student_factory=factory,
                options=RESPAOptions(0.02),
                outer_steps=1,
            )
        assert not closed
        assert next(generator) == "still usable"
    finally:
        generator.close()


@pytest.mark.parametrize("kind", ["coroutine", "async_generator"])
def test_chained_error_cleanup_preserves_suspended_caller_async_tasks(kind):
    async def main():
        closed = []
        previous = []
        resume = asyncio.Event()

        async def coroutine():
            try:
                try:
                    raise ValueError("previous unrelated error")
                except ValueError as error:
                    previous.append(error)
                await resume.wait()
                return "still usable"
            finally:
                closed.append(True)

        async def async_generator():
            try:
                try:
                    raise ValueError("previous unrelated error")
                except ValueError as error:
                    yield error
                yield "still usable"
            finally:
                closed.append(True)

        def factory():
            raise RuntimeError("factory failed") from previous[0]

        if kind == "coroutine":
            source = asyncio.create_task(coroutine())
            await asyncio.sleep(0)
        else:
            source = async_generator()
            previous.append(await anext(source))
        try:
            with pytest.raises(RuntimeError, match="factory failed"):
                run_respa(
                    _atom(),
                    teacher_factory=factory,
                    student_factory=factory,
                    options=RESPAOptions(0.02),
                    outer_steps=1,
                )
            assert not closed
            if kind == "coroutine":
                resume.set()
                assert await source == "still usable"
            else:
                assert await anext(source) == "still usable"
        finally:
            if kind == "coroutine":
                resume.set()
                await asyncio.gather(source, return_exceptions=True)
            else:
                await source.aclose()

    asyncio.run(main())
