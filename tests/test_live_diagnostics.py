import pytest

from liminal.core.types import SO101_JOINTS
from liminal.policy.live_diagnostics import ChunkDiagnostics


def test_distinguishes_calibrated_clipping_controller_adjustment_and_actual_motion():
    metrics = ChunkDiagnostics([0] * 6, 10)
    requested, submitted, issued = [12] * 6, [10] * 6, [1] * 6
    metrics.record(requested, submitted, issued, 0.01, {SO101_JOINTS[0]: -0.6})
    metrics.record(issued, issued, issued, 0.03, {SO101_JOINTS[0]: 0.2})
    result = metrics.summary([0.5] * 6, 10.1)
    assert result['command_phase_hz'] == pytest.approx(20)
    assert result['ack_mean_ms'] == 20
    assert result['ack_max_ms'] == 30
    assert set(result['calibrated_clip_counts'].values()) == {1}
    assert set(result['controller_adjustment_counts'].values()) == {1}
    assert set(result['max_controller_adjustment'].values()) == {9}
    assert set(result['measured_delta'].values()) == {0.5}
    assert result['sampled_peak_load_fraction'][SO101_JOINTS[0]] == 0.6
    assert requested == [12] * 6 and submitted == [10] * 6 and issued == [1] * 6


def test_missing_load_and_empty_chunk_are_not_reported_as_measured_zero():
    metrics = ChunkDiagnostics([0] * 6, 10)
    result = metrics.summary([0] * 6, 10)
    assert result['command_phase_hz'] is None
    assert result['ack_mean_ms'] is None
    assert result['sampled_peak_load_fraction'] == {}
