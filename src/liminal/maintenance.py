"""The pen demo: five fixed steps, five photo checks, no planner or replanning."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

COLORS = ('red', 'green', 'blue', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown')


@dataclass(frozen=True)
class Step:
    title: str
    arm: str
    prompts: tuple[str, ...]
    checks: tuple[str, ...]


def steps(color: str) -> tuple[Step, ...]:
    if color not in COLORS:
        raise ValueError(f'Choose a faulty pen color: {", ".join(COLORS)}')
    return (
        Step('Open the box', 'left', ('Open the box lid',), ('lid_fully_open_and_resting',)),
        Step('Move and hold the cable', 'left', ('Move the cable aside and hold it',),
             ('cable_held_clear', 'all_pens_visible_and_reachable')),
        Step('Remove the faulty pen', 'right', (f'Pick up the {color} pen and put it in the faulty cup',),
             ('cable_held_clear', 'faulty_pen_in_faulty_cup', 'original_slot_empty')),
        Step('Insert the replacement', 'right',
             ('Pick up a pen from the replacement cup and place it in the empty slot',),
             ('cable_held_clear', 'replacement_seated_in_original_slot', 'right_arm_retracted')),
        Step('Release and close', 'left', ('Release the cable', 'Close the box lid'), ('lid_closed',)),
    )


class PhotoVerifier:
    """Claude only judges photos; it never chooses a step or generates a robot command."""

    def check(self, step: Step, color: str, current: bytes, reference: bytes | None = None) -> dict:
        import anthropic

        from liminal.settings import claude_model

        content = []
        for label, image in [('Previous checkpoint (reference only)', reference), ('CURRENT checkpoint', current)]:
            if image:
                content += [{'type': 'text', 'text': label}, {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': base64.b64encode(image).decode()}}]
        content.append({'type': 'text', 'text': json.dumps({
            'task': 'Replace the faulty pen', 'faulty_color': color, 'step': step.title,
            'required_checks': step.checks,
        })})
        response = anthropic.Anthropic(timeout=30, max_retries=0).messages.create(
            model=claude_model(), max_tokens=700,
            system='You independently verify a robot maintenance checkpoint from photos. '
            'The hinged box contains flat pens in individual slots, with a cable held aside by the left arm. '
            'Outside: the replacement cup initially contains pens; the faulty cup is initially empty. '
            'Every check must be visibly supported in CURRENT. Compare the previous checkpoint for the original slot. '
            'Never substitute colors. No evidence or occlusion means null, never true. '
            'Do not trust a requested action as evidence it happened. '
            'Return JSON only: {"checks": {each_required_check: true|false|null}, '
            '"reason": "short visual explanation"}.',
            messages=[{'role': 'user', 'content': content}])
        raw = ''.join(block.text for block in response.content if block.type == 'text').strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        result = json.loads(raw)
        if (not isinstance(result, dict) or not isinstance(result.get('checks'), dict)
                or set(result['checks']) != set(step.checks)
                or any(v is not None and type(v) is not bool for v in result['checks'].values())
                or not isinstance(result.get('reason'), str)):
            raise ValueError('Invalid checkpoint response; step remains unverified')
        return result


@dataclass
class WorldState:
    """Only camera verification advances the fixed sequence."""
    completed: list[str] = field(default_factory=list)
    last_check: dict = field(default_factory=dict)


class Maintenance:
    def __init__(self, folder: Path, color: str = 'red', robot=None, verifier=None):
        self.steps = steps(color)
        self.color, self.robot = color, robot
        self.verifier = verifier or PhotoVerifier()
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.world = WorldState()
        self.index = 0
        self.status = 'ready'
        self.busy = False
        self.cancel = threading.Event()
        self.attempts = [0] * 5
        self.last_photo = 0
        self.log('job', color=color, mode='robot' if robot else 'photo review; no robot connected')

    def state(self) -> dict:
        return {'color': self.color, 'step': min(5, self.index + 1), 'status': self.status, 'busy': self.busy,
                'robot_connected': self.robot is not None, 'steps': [asdict(s) for s in self.steps],
                'world': asdict(self.world), 'folder': str(self.folder),
                'checkpoint': self.last_photo,
                'left_hold': self.robot.left_hold if self.robot else False}

    def log(self, event: str, **data):
        with (self.folder / 'log.txt').open('a') as f:
            f.write(json.dumps({'event': event, **data}) + '\n')
        (self.folder / 'state.json').write_text(json.dumps(self.state(), indent=2))

    async def run_step(self):
        if not self.robot:
            raise ValueError('Photo review only: upload a real checkpoint photo, or start with --robot')
        await self._step(None)

    async def run_job(self):
        """One operator start; advance automatically through verified checkpoints."""
        while self.index < 5 and self.status not in ('stopped', 'complete'):
            await self.run_step()
            if self.status != 'ready':
                break

    async def review_photo(self, data: bytes):
        if self.robot:
            raise ValueError('Robot mode captures its own checkpoint; uploaded photos cannot advance the robot')
        await self._step(data)

    async def _step(self, photo: bytes | None):
        if self.busy or self.status in ('complete', 'stopped'):
            raise ValueError('Job is busy, complete or stopped')
        self.busy, self.status = True, 'running'
        self.cancel.clear()
        step = self.steps[self.index]
        self.attempts[self.index] += 1
        self.log('step_start', step=self.index + 1, arm=step.arm, prompts=step.prompts)
        try:
            if self.robot:
                await asyncio.to_thread(self.robot.perform, self.index, step, self.cancel)
                if self.cancel.is_set():
                    return
                photo = await asyncio.to_thread(self.robot.photo)
            from PIL import Image

            image = Image.open(io.BytesIO(photo))
            image.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
            photo = buffer.getvalue()
            path = self.folder / f'checkpoint-{self.index + 1}.png'
            path.write_bytes(photo)
            (self.folder / f'step-{self.index + 1}-attempt-{self.attempts[self.index]}.png').write_bytes(photo)
            self.last_photo = self.index + 1
            previous = self.folder / f'checkpoint-{self.index}.png'
            result = await asyncio.to_thread(self.verifier.check, step, self.color, photo,
                                             previous.read_bytes() if previous.exists() else None)
            if self.cancel.is_set():
                return  # A late model response must not advance a stopped job.
            self.world.last_check = result
            passed = all(result['checks'].get(name) is True for name in step.checks)
            self.log('checkpoint', step=self.index + 1, passed=passed, **result)
            if passed:
                self.world.completed.append(step.title)
                self.index += 1
            self.status = 'complete' if self.index == 5 else ('ready' if passed else 'paused')
        except Exception as exc:
            if not self.cancel.is_set():
                self.status = 'paused'
            self.world.last_check = {'error': str(exc)}
            self.log('error', message=str(exc))
        finally:
            self.busy = False
            self.log('step_end', status=self.status)

    def stop(self):
        self.cancel.set()
        self.status = 'stopped'
        if self.robot:
            self.robot.stop()
        self.log('stop', reason='Hold both arms; discard late actions and model responses')
