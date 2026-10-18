"""Atomic roles, arbitrary ordering, fresh evidence and fail-stop behavior."""
import asyncio
from itertools import permutations
from pathlib import Path

import pytest
import yaml

from liminal.core.types import SkillResult, Status
from liminal.orchestration import run_task
from liminal.orchestration.component_assembly import COMPONENTS, DEFINITIONS, assembly_task
from liminal.perception.types import InspectionProfile, SceneAssessment


class Scene:
    def __init__(self, ctx, values):
        self.ctx, self.values = ctx, values

    async def inspect_until_known(self, request, **kwargs):
        values = {**self.values, 'components_assembled': all(
            self.values.get(f'{item}_in_slot', False) for item in COMPONENTS)}
        assessment = SceneAssessment.from_model_output(request, {'predicates': {
            key: {'value': values.get(key, False), 'evidence': 'independent fixture sensor'}
            for key in request.predicates}})
        for item in assessment.observations.values():
            self.ctx.world.observe(item.predicate, item.value, captured_at=assessment.captured_at, source='fixture')
        return assessment


class AtomicBackend:
    def __init__(self, values, fail=None, lose_lid=False):
        self.values, self.fail, self.lose_lid = values, fail, lose_lid
        self.calls = []

    async def execute(self, skill, ctx):
        self.calls.append((skill.arm, skill.name))
        if skill.name != self.fail:
            self.values[skill.success] = True
        if self.lose_lid and skill.arm == 'left':
            self.values['lid_open'] = False
            ctx.world.observe('lid_open', False, source='fixture')
            await asyncio.sleep(0.03)
        from liminal.perception.types import InspectionRequest
        await ctx.inspector.inspect_until_known(InspectionRequest(predicates=[skill.success]))
        return SkillResult(Status.SUCCESS, skill)


def run(make_ctx, order=COMPONENTS, preplaced=(), fail=None, lose_lid=False):
    values = {f'{item}_in_slot': True for item in preplaced}
    backend = AtomicBackend(values, fail, lose_lid)
    ctx = make_ctx(backend)
    ctx.inspection = InspectionProfile(definitions=DEFINITIONS)
    ctx.inspector = Scene(ctx, values)
    task = assembly_task(order)
    # Bound negative verification cases without waiting production timeouts.
    def shorten(node):
        if isinstance(node, dict):
            if 'skill' in node:
                node['skill']['verify_timeout_s'] = 0.01
            for value in node.values():
                shorten(value)
        elif isinstance(node, list):
            for value in node:
                shorten(value)
    shorten(task.raw)
    from liminal.orchestration import parse_task
    report = asyncio.run(run_task(parse_task(task.raw), ctx))
    return report, backend


@pytest.mark.parametrize('order', list(permutations(COMPONENTS)))
def test_every_order_uses_correct_arms(make_ctx, order):
    report, backend = run(make_ctx, order)
    assert report.goal_verified
    assert backend.calls == [('right', 'open_box'), *[('left', f'put_{item}_in_slot') for item in order]]


def test_fresh_preplaced_items_are_skipped(make_ctx):
    report, backend = run(make_ctx, preplaced=('red_pen',))
    assert report.goal_verified
    assert ('left', 'put_red_pen_in_slot') not in backend.calls


@pytest.mark.parametrize('failed', ['open_box', 'put_green_pen_in_slot', 'put_red_pen_in_slot',
                                    'put_yellow_boxcutter_in_slot'])
def test_unverified_success_stops_without_retry(make_ctx, failed):
    report, backend = run(make_ctx, fail=failed)
    assert not report.goal_verified
    assert backend.calls[-1][1] == failed
    assert sum(name == failed for _, name in backend.calls) == 1


def test_lid_loss_stops_placement(make_ctx):
    report, backend = run(make_ctx, lose_lid=True)
    assert not report.goal_verified
    assert len(backend.calls) == 2


@pytest.mark.parametrize('order', [(), ('red_pen',)*3, 'red_pen', ('green_pen', 'red_pen', 'cable')])
def test_invalid_orders_rejected(order):
    with pytest.raises(ValueError):
        assembly_task(order)


def test_shipped_task_and_registry_match():
    assert yaml.safe_load(Path('configs/tasks/assemble_components.yaml').read_text()) == assembly_task().raw
    cfg = yaml.safe_load(Path('configs/policy/showcase_registry.yaml').read_text())
    prompts = {}

    def collect(node):
        if isinstance(node, dict):
            if 'skill' in node:
                prompts[node['skill']['name']] = node['skill']['prompt']
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(assembly_task().raw)
    assert prompts == {
        'open_box': 'open the box by pulling its handle',
        'put_green_pen_in_slot': 'put the green pen in the green slot',
        'put_yellow_boxcutter_in_slot': 'put the yellow cutter in the yellow slot',
        'put_red_pen_in_slot': 'put the red pen in its slot in the cardboard box',
    }
    assert cfg['execute_steps'] == 16
    assert set(cfg['skill_checkpoints']) == {'open_box', *[f'put_{item}_in_slot' for item in COMPONENTS]}
    for skill, checkpoint in cfg['skill_checkpoints'].items():
        assert cfg['checkpoints'][checkpoint]['targets'] == ['right' if skill == 'open_box' else 'left']
