"""Pure absolute-step planning; no arrays, device imports or filesystem effects."""

from .schemas import ZarrCapturePolicy


def next_chunk_length(
    *, state_step: int, remaining_steps: int, chunk_steps: int, policy: ZarrCapturePolicy
) -> int:
    if state_step < 0 or remaining_steps <= 0 or chunk_steps <= 0:
        raise ValueError("invalid capture planning horizon")
    distances = [remaining_steps, chunk_steps]
    for interval in (policy.checkpoint_interval, policy.snapshot_interval):
        if interval is not None:
            distances.append(interval - state_step % interval)
    return min(distances)


def expected_capture_steps(
    *, initial_step: int, total_steps: int, interval: int | None, on_completion: bool = False
) -> tuple[int, ...]:
    if initial_step < 0 or total_steps <= 0:
        raise ValueError("invalid capture horizon")
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("interval must be a positive integer or None")
    end = initial_step + total_steps
    values = (
        list(range((initial_step // interval + 1) * interval, end + 1, interval))
        if interval
        else []
    )
    if on_completion and (not values or values[-1] != end):
        values.append(end)
    return tuple(values)
