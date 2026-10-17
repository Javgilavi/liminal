"""MemER-style keyframe memory for the planner context.

MemER keeps a window of recent frames plus a few selected historical keyframes. A high-level model
nominates keyframes from recent history, nominations that lie close together in time are clustered,
and each cluster is represented by its median frame. This module adapts that idea to the harness:

* ``event``     - nominations come from verified ledger events: pre/post frames of verified attempts and
                  the frame after a failed attempt (the rule the episode runner always used).
* ``nominated`` - nominations come from the planner proposal (``keyframe_nominations``). The proposal
                  validator only lets through frame ids the planner was shown, and the result is
                  advisory: a nomination can add context, never evidence.
* ``none``      - ablation baseline: recent frames and the goal reference only.

In every mode the goal reference frames survive. Everything is bounded (``max_keyframes`` clusters,
the store's recent-frame window and selected-frame limit), persisted with its reason, and restored on
resume. Frames are images only: no keyframe makes an obligation true.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .store import FrameRecord, MemoryStore

KEYFRAME_MODES = ("event", "nominated", "none")
REFERENCE_PREFIX = "goal_reference"
DEFAULT_MAX_KEYFRAMES = 4
NOMINATION_KIND = "keyframe_nomination"
SELECTED_KIND = "keyframe_selected"
REJECTED_KIND = "keyframe_rejected"
MAX_REJECTED_KEPT = 16


def plain(value: Any) -> Any:
    """Detach a frozen store payload (MappingProxyType / tuples) into plain JSON-able values."""
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Nomination:
    frame_id: str
    reason: str
    source: str  # "event" | "planner"
    event: str = ""
    seq: int = 0  # position on the run's frame time axis
    revision: int | None = None
    attempt_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "reason": self.reason, "source": self.source, "event": self.event,
                "seq": self.seq, "revision": self.revision, "attempt_id": self.attempt_id}


@dataclass(frozen=True)
class Cluster:
    keyframe: Nomination  # the median member represents the cluster
    members: tuple[Nomination, ...]

    @property
    def reason(self) -> str:
        reasons = list(dict.fromkeys(m.reason for m in self.members))
        return "; ".join(reasons)[:240]


def cluster_nominations(nominations: Iterable[Nomination], gap: int = 1) -> list[Cluster]:
    """Group nominations whose frames are at most ``gap`` frames apart; keep each group's median frame.

    One entry per distinct frame (reasons merge). Clusters are returned oldest first.
    """
    if gap < 0:
        raise ValueError("cluster gap must be non-negative")
    merged: dict[str, Nomination] = {}
    for item in nominations:
        previous = merged.get(item.frame_id)
        if previous is None:
            merged[item.frame_id] = item
        elif item.reason not in previous.reason:
            merged[item.frame_id] = Nomination(
                previous.frame_id, f"{previous.reason}; {item.reason}"[:240], previous.source, previous.event,
                previous.seq, previous.revision, previous.attempt_id)
    ordered = sorted(merged.values(), key=lambda n: (n.seq, n.frame_id))
    groups: list[list[Nomination]] = []
    for item in ordered:
        if groups and item.seq - groups[-1][-1].seq <= gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [Cluster(group[(len(group) - 1) // 2], tuple(group)) for group in groups]


class KeyframeMemory:
    """Per-run keyframe selection over a ``MemoryStore``; state is restored from the store on construction."""

    def __init__(self, store: MemoryStore, run_id: str, mode: str = "event", *,
                 max_keyframes: int = DEFAULT_MAX_KEYFRAMES, gap: int = 1):
        if mode not in KEYFRAME_MODES:
            raise ValueError(f"unknown keyframe mode {mode!r} (expected one of {KEYFRAME_MODES})")
        if max_keyframes < 0 or gap < 0:
            raise ValueError("max_keyframes and gap must be non-negative")
        self.store, self.run_id, self.mode = store, run_id, mode
        self.max_keyframes, self.gap = max_keyframes, gap
        self._nominations: list[Nomination] = [
            Nomination(**{k: v for k, v in plain(n.payload).items()})
            for n in store.notes(run_id, NOMINATION_KIND)]
        self._counter = len(self._nominations)
        self._rejected_counter = max(
            (int(n.note_id[1:]) for n in store.notes(run_id, REJECTED_KIND) if n.note_id[1:].isdigit()), default=0)

    # -- queries ----------------------------------------------------------------------------

    @staticmethod
    def is_reference(frame: FrameRecord) -> bool:
        return str(frame.event).startswith(REFERENCE_PREFIX)

    def references(self) -> list[FrameRecord]:
        return [f for f in self.store.selected_frames(self.run_id) if self.is_reference(f)]

    def _alive(self, nominations: Sequence[Nomination]) -> list[Nomination]:
        alive = []
        for item in nominations:
            try:
                self.store.get_frame(self.run_id, item.frame_id)
            except KeyError:
                continue  # evicted; nothing left to show
            alive.append(item)
        return alive

    def keyframes(self) -> list[Cluster]:
        """Current representative keyframes, oldest first, bounded by ``max_keyframes``."""
        if self.mode == "none" or self.max_keyframes == 0:
            return []
        alive = self._alive([n for n in self._nominations
                             if n.source == ("event" if self.mode == "event" else "planner")])
        if self.mode == "event":
            # Pre/post pairs must stay distinct, so events are not merged: just the newest N frames.
            clusters = [Cluster(n, (n,)) for n in {n.frame_id: n for n in alive}.values()]
            clusters.sort(key=lambda c: (c.keyframe.seq, c.keyframe.frame_id))
        else:
            clusters = cluster_nominations(alive, self.gap)
        return clusters[-self.max_keyframes:]

    # -- nominations ------------------------------------------------------------------------

    def _frame_seq(self, frame: FrameRecord) -> int:
        seq = frame.metadata.get("seq")
        return int(seq) if isinstance(seq, int) and not isinstance(seq, bool) else 0

    def _record(self, frame: FrameRecord, reason: str, source: str, event: str, attempt_id: str | None) -> Nomination:
        revision = frame.metadata.get("revision")
        nomination = Nomination(frame.frame_id, reason[:240], source, event, self._frame_seq(frame),
                                revision if isinstance(revision, int) and not isinstance(revision, bool) else None,
                                attempt_id)
        self._counter += 1
        self.store.put_note(self.run_id, NOMINATION_KIND, f"n{self._counter:04d}", nomination.as_dict())
        self._nominations.append(nomination)
        return nomination

    def note_event(self, frame_ids: Iterable[str], event: str, reason: str, attempt_id: str | None = None) -> list[str]:
        """Event rule: a verified-ledger moment (pre/post/failure) points at these frames. ``event`` mode only."""
        if self.mode != "event":
            return []
        recorded = []
        for frame_id in dict.fromkeys(frame_ids):
            try:
                frame = self.store.get_frame(self.run_id, frame_id)
            except KeyError:
                continue
            if self.is_reference(frame):
                continue
            self._record(frame, f"{event}: {reason}" if reason else event, "event", event, attempt_id)
            recorded.append(frame_id)
        return recorded

    def accept_planner_nominations(self, nominations: Iterable[dict[str, Any]], *,
                                   attempt_id: str | None = None) -> list[str]:
        """Record planner nominations already checked against the frames the planner was shown.

        Advisory only: a frame that has since been evicted or is a goal reference is skipped, never an error.
        """
        if self.mode != "nominated":
            return []
        recorded = []
        for item in nominations:
            frame_id = str(item.get("frame_id", ""))
            try:
                frame = self.store.get_frame(self.run_id, frame_id)
            except KeyError:
                self._reject(frame_id, "frame no longer stored")
                continue
            if self.is_reference(frame):
                self._reject(frame_id, "goal reference frames are always kept")
                continue
            self._record(frame, str(item.get("reason", "")) or "planner nomination", "planner", "planner_nomination",
                         attempt_id)
            recorded.append(frame_id)
        return recorded

    def _reject(self, frame_id: str, why: str) -> None:
        rejected = self.store.notes(self.run_id, REJECTED_KIND)
        for stale in rejected[: max(0, len(rejected) - MAX_REJECTED_KEPT + 1)]:
            self.store.delete_note(self.run_id, REJECTED_KIND, stale.note_id)
        self._rejected_counter += 1
        self.store.put_note(self.run_id, REJECTED_KIND, f"r{self._rejected_counter}",
                            {"frame_id": frame_id, "reason": why})

    def record_rejected(self, rejected: Iterable[dict[str, Any]]) -> None:
        """Persist nominations the validator refused (unknown frame id, malformed), for inspection."""
        for item in rejected:
            self._reject(str(item.get("frame_id", "")), str(item.get("error", "rejected")))

    # -- selection --------------------------------------------------------------------------

    def apply(self, reason: str = "episode: goal reference + keyframes",
              add_references: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Replace the store's selected frames with references + current keyframes; persist the reasons.

        ``add_references`` names newly seen goal-reference frames to start keeping. Existing references
        are always carried over, so nothing ever evicts them.
        """
        refs = self.references()
        known = {r.frame_id for r in refs}
        for frame_id in dict.fromkeys(add_references):
            if frame_id not in known:
                try:
                    refs.append(self.store.get_frame(self.run_id, frame_id))
                except KeyError:
                    continue
        clusters = self.keyframes()
        wanted: list[str] = []
        entries: list[dict[str, Any]] = []
        for ref in refs:
            wanted.append(ref.frame_id)
            entries.append({"frame_id": ref.frame_id, "reason": "goal reference", "event": ref.event,
                            "source": "reference", "seq": self._frame_seq(ref), "revision": None, "cluster_size": 1})
        for cluster in clusters:
            key = cluster.keyframe
            wanted.append(key.frame_id)
            entries.append({"frame_id": key.frame_id, "reason": cluster.reason, "event": key.event,
                            "source": key.source, "seq": key.seq, "revision": key.revision,
                            "cluster_size": len(cluster.members),
                            "cluster_frames": [m.frame_id for m in cluster.members]})
        wanted = list(dict.fromkeys(wanted))
        self.store.select_frames(self.run_id, wanted, reason=reason)
        keep = {e["frame_id"] for e in entries}
        for stale in self.store.notes(self.run_id, SELECTED_KIND):
            if stale.note_id not in keep:
                self.store.delete_note(self.run_id, SELECTED_KIND, stale.note_id)
        for entry in entries:
            self.store.put_note(self.run_id, SELECTED_KIND, entry["frame_id"], entry)
        return entries

    def context_frames(self, cap: int, exclude: Iterable[str] = ()) -> list[FrameRecord]:
        """Historical frames for the planner: references first, then keyframes, then the newest recent frames.

        At most ``cap`` frames; references are never displaced by keyframes or recents. Chronological order.
        """
        skip = set(exclude)
        chosen: list[FrameRecord] = []
        seen = set(skip)

        def take(frames: Iterable[FrameRecord]) -> None:
            for frame in frames:
                if len(chosen) >= cap:
                    return
                if frame.frame_id not in seen:
                    seen.add(frame.frame_id)
                    chosen.append(frame)

        take(self.references())
        take(self.store.get_frame(self.run_id, c.keyframe.frame_id) for c in reversed(self.keyframes()))
        take(reversed(self.store.recent_frames(self.run_id)))
        chosen.sort(key=lambda f: (f.captured_at, f.frame_id))
        return chosen

    # -- inspection -------------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "max_keyframes": self.max_keyframes, "cluster_gap": self.gap,
            "references": [f.frame_id for f in self.references()],
            "selected": [plain(n.payload) for n in self.store.notes(self.run_id, SELECTED_KIND)],
            "nominations": [n.as_dict() for n in self._nominations],
            "rejected": [plain(n.payload) for n in self.store.notes(self.run_id, REJECTED_KIND)],
        }
