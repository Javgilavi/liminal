"""Observational rollout metrics; never modifies commands or control settings."""

from liminal.core.types import SO101_JOINTS


class ChunkDiagnostics:
    def __init__(self, initial: list[float], started: float):
        self.initial = list(initial)
        self.started = started
        self.count = 0
        self.ack_total = 0.0
        self.ack_max = 0.0
        self.clipped = dict.fromkeys(SO101_JOINTS, 0)
        self.limited = dict.fromkeys(SO101_JOINTS, 0)
        self.peak_load: dict[str, float] = {}
        self.max_controller_adjustment = dict.fromkeys(SO101_JOINTS, 0.0)

    def record(self, requested, submitted, committed, ack_seconds, loads) -> None:
        self.count += 1
        self.ack_total += ack_seconds
        self.ack_max = max(self.ack_max, ack_seconds)
        for joint, request, target, issued in zip(SO101_JOINTS, requested, submitted, committed, strict=True):
            self.clipped[joint] += int(abs(request - target) > 1e-5)
            self.limited[joint] += int(abs(target - issued) > 1e-5)
            self.max_controller_adjustment[joint] = max(self.max_controller_adjustment[joint], abs(target - issued))
        for joint, value in loads.items():
            if value is not None:
                self.peak_load[joint] = max(self.peak_load.get(joint, 0.0), abs(value))

    def summary(self, measured, finished) -> dict:
        elapsed = finished - self.started
        return {"commands": self.count, "command_phase_s": round(elapsed, 3),
                "command_phase_hz": round(self.count / elapsed, 2) if elapsed > 0 else None,
                "ack_mean_ms": round(1000 * self.ack_total / self.count, 2) if self.count else None,
                "ack_max_ms": round(1000 * self.ack_max, 2),
                "calibrated_clip_counts": self.clipped, "controller_adjustment_counts": self.limited,
                "max_controller_adjustment": self.max_controller_adjustment,
                "measured_delta": {j: round(q - start, 3)
                                   for j, q, start in zip(SO101_JOINTS, measured, self.initial, strict=True)},
                "sampled_peak_load_fraction": self.peak_load}
