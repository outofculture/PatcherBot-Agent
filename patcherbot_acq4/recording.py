"""Lossless phase event recording alongside ACQ4's native patch log.

No sensor values are interpolated or assigned the timestamp of another sensor.
Each frame keeps its acquisition metadata and a separate monotonically timed
phase observation; the offline exporter can refuse ambiguous alignment.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from acq4.util import Qt
from acq4.util.multipatch_log_recorder import MultiPatchLogRecorder

from .models import sha256


def _encode(value, frame_dir: Path, prefix: str):
    if isinstance(value, np.ndarray):
        name = f"{prefix}-{uuid4().hex}.npy"
        np.save(frame_dir / name, value, allow_pickle=False)
        return {"array_file": f"arrays/{name}", "shape": value.shape, "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return _encode(value.item(), frame_dir, prefix)
    if isinstance(value, dict):
        return {str(k): _encode(v, frame_dir, prefix) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, frame_dir, prefix) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot record value of type {type(value).__name__}")


class AttemptRecorder:
    """Write one phase/event stream and a native ACQ4 MultiPatch log per attempt."""

    def __init__(self, manager, pipette, *, config: dict, models: dict[str, str | Path]):
        parent = manager.getCurrentDir()
        if parent is None:
            raise RuntimeError("Select an ACQ4 storage directory before recording a patch attempt")
        self.directory = parent.mkdir("PatcherBot", autoIncrement=True)
        self.path = Path(self.directory.name())
        self.arrays = self.path / "arrays"
        self.arrays.mkdir()
        self._lock = threading.RLock()
        self._closed = False
        self._seq = 0
        self._pipette = pipette
        self._events = (self.path / "phase_events.jsonl").open("x", encoding="utf-8")
        provenance = {
            "schema": "patcherbot_acq4_attempt_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "clock_anchor": {"wall_time_s": time.time(), "monotonic_s": time.monotonic()},
            "pipette": pipette.name(),
            "module_config": config,
            "model_artifacts": {
                name: {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256(path) if Path(path).is_file() else None,
                    "available": Path(path).is_file(),
                }
                for name, path in models.items()
            },
        }
        (self.path / "manifest.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._native = MultiPatchLogRecorder(
            self.directory, pipettes=(pipette,), record_full_test_pulses=True
        )
        pipette.sigNewEvent.connect(self._on_device_event, type=Qt.Qt.DirectConnection)

    def record(self, phase: str, observation: dict, calculations: dict, action: dict):
        """Record explicit observation/action evidence with its original clock basis."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Attempt recorder is closed")
            seq = self._seq
            self._seq += 1
            row = {
                "sequence": seq,
                "recorded_monotonic_s": time.monotonic(),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "observation": _encode(observation, self.arrays, f"{seq}-obs"),
                "calculations": _encode(calculations, self.arrays, f"{seq}-calc"),
                "action": _encode(action, self.arrays, f"{seq}-action"),
            }
            self._events.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self._events.flush()

    def operator_result(self, phase: str, outcome: str):
        if outcome not in ("success", "abort"):
            raise ValueError("Operator outcome must be success or abort")
        self.record(phase, {}, {"operator_outcome": outcome}, {})

    def _on_device_event(self, pipette, event):
        """Retain wall-clock ACQ4 events with a paired monotonic receive time."""
        with self._lock:
            if self._closed:
                return
            seq = self._seq
            self._seq += 1
            payload = dict(event)
            # The complete waveform is already stored in TestPulses.hdf5 by the
            # native recorder. Its Python object is not a JSON value.
            full_pulse = payload.pop("full_test_pulse", None)
            row = {
                "sequence": seq,
                "recorded_monotonic_s": time.monotonic(),
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "device_event": _encode(payload, self.arrays, f"{seq}-device"),
                "full_test_pulse_in_native_sidecar": full_pulse is not None,
            }
            self._events.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self._events.flush()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pipette.sigNewEvent.disconnect(self._on_device_event)
            self._native.stop()
            self._events.close()
