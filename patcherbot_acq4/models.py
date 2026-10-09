"""PatcherBot's distinct microscopy and learned phase inference contracts.

Models are loaded only on demand; missing weights never turn into a different model.
The detector architectures are MIT-licensed copies of the inference-only model
factories in Neuron_Detection (see model_runtime/LICENSE). The policy input
transformations are adapted from PatcherBotAgentR.py and use its custom
robomimic environment when its author-supplied fork is installed.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import threading
from pathlib import Path
from typing import Mapping

import numpy as np


_POLICY_CLASSES = {
    "find_pipette": "PipetteFinder",
    "hunt": "CellHunter",
    "gigaseal": "GigaSealer",
    "break_in": "Burglar",
}


def sha256(path: str | Path) -> str:
    """Hash a provided model artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(value: str | Path | None, purpose: str) -> Path:
    if not value:
        raise FileNotFoundError(f"{purpose} requires an explicit checkpoint path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} checkpoint not found: {path}")
    return path


def _camera_uint8(image: np.ndarray, input_range: tuple[float, float] | None) -> np.ndarray:
    """Map raw 16-bit sensor intensities using explicit rig-calibrated bounds."""
    frame = np.asarray(image)
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype != np.uint16:
        raise ValueError(f"Unsupported camera pixels {frame.dtype}; expected uint8 or uint16")
    if input_range is None:
        raise ValueError("16-bit camera frames require detector.inputRange [lower, upper]")
    lower, upper = input_range
    scaled = frame.astype(np.float32)
    scaled -= lower
    scaled *= 255.0 / (upper - lower)
    np.clip(scaled, 0.0, 255.0, out=scaled)
    return scaled.astype(np.uint8)


class PatcherBotDetector:
    """Load the project's 384px tip/defocus model and optional cell detector.

    No inference exception is converted into an apparent negative detection:
    an unavailable model cannot safely authorize movement.
    """

    def __init__(
        self,
        pipette_checkpoint: str | Path,
        *,
        cell_checkpoint: str | Path | None = None,
        cell_model: str = "segformer",
        cell_encoder_config: str | Path | None = None,
        device: str | None = None,
        threshold: float = 0.001,
        cuda_graph: bool = True,
        input_range: tuple[float, float] | None = None,
    ):
        if input_range is not None:
            if len(input_range) != 2:
                raise ValueError("detector.inputRange must contain lower and upper sensor counts")
            lower, upper = map(float, input_range)
            if not all(math.isfinite(x) for x in (lower, upper)) or not 0 <= lower < upper <= 65535:
                raise ValueError("detector.inputRange must be finite 16-bit camera counts")
            input_range = (lower, upper)
        self.input_range = input_range
        self.pipette_checkpoint = _required_file(pipette_checkpoint, "PipetteDetector4")
        from .model_runtime.pipette_factory import PDModelFactory

        self.pipette_sha256 = sha256(self.pipette_checkpoint)
        self._pipette = PDModelFactory.create(
            model_type="unet_mobilenetv4_boxes",
            model_path=self.pipette_checkpoint,
            device=device,
            threshold=threshold,
            cuda_graph=cuda_graph,
        )
        self._cell = None
        self.cell_checkpoint = None
        self.cell_sha256 = None
        if cell_checkpoint is not None:
            self.cell_checkpoint = _required_file(cell_checkpoint, "CellDetector2")
            from .model_runtime.cell_factory import CDModelFactory

            self.cell_sha256 = sha256(self.cell_checkpoint)
            self._cell = CDModelFactory.create(
                model_type=cell_model,
                model_path=self.cell_checkpoint,
                encoder_config_path=cell_encoder_config,
                device=device,
            )

    def require(self, phase: str):
        """Preflight the detector heads required before entering a visual phase."""
        if phase in ("approach_cell", "hunt") and self._cell is None:
            raise FileNotFoundError(
                f"{phase} Adaptive mode requires a configured CellDetector2 checkpoint"
            )
        if self._pipette is None:
            raise RuntimeError(f"{phase} requires a loaded PipetteDetector4")

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        return _camera_uint8(image, self.input_range)

    def detect_pipette_details(self, image: np.ndarray, *, include_mask: bool = False) -> dict:
        import cv2

        frame = self.prepare_image(image)
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                frame = frame[:, :, 0]
            elif frame.shape[2] in (3, 4):
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
                )
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError("PipetteDetector4 requires a nonempty grayscale/BGR frame")
        if frame.dtype != np.uint8:
            raise ValueError("PipetteDetector4 requires normalized uint8 camera pixels")
        result = self._pipette.predict(frame, native_mask=include_mask)
        tip = result.get("tip_xy")
        if tip is not None:
            tip = tuple(float(v) for v in tip)
            if len(tip) != 2 or not np.isfinite(tip).all():
                raise ValueError("PipetteDetector4 returned an invalid tip coordinate")
        return {
            "xy": tip,
            "z_um": result.get("z_um"),
            "box_confidence": result.get("box_confidence"),
            "z_confidence": result.get("z_confidence"),
            "mask": result.get("mask"),
        }

    def detect_cells(self, image: np.ndarray) -> list[tuple[int, int, float]]:
        if self._cell is None:
            raise RuntimeError("Adaptive cell detection requires a CellDetector2 checkpoint")
        return self._cell.predict(self.prepare_image(image))


class PolicyRuntime:
    """Load only the original phase policies and preserve their input transforms.

    The custom robomimic fork must contain EnvPatcherOnline. Checkpoints and
    custom fork are external artifacts; error messages name the missing item.
    """

    def __init__(self, checkpoints: Mapping[str, str | Path], *, robomimic_root: str | Path):
        self.checkpoints = dict(checkpoints)
        self.robomimic_root = Path(robomimic_root).expanduser().resolve()
        self._agents = {}
        self._locks = {name: threading.Lock() for name in _POLICY_CLASSES}

    def require(self, phase: str):
        if phase not in _POLICY_CLASSES:
            raise ValueError(f"No deployed PatcherBot policy for phase {phase!r}")
        checkpoint = _required_file(self.checkpoints.get(phase), f"{phase} policy")
        env = self.robomimic_root / "robomimic" / "envs" / "env_patcher_online.py"
        if not env.is_file():
            raise FileNotFoundError(
                f"{phase} policy requires the author's robomimic fork with EnvPatcherOnline: {env}"
            )
        return checkpoint

    def _agent(self, phase: str):
        checkpoint = self.require(phase)
        if phase not in self._agents:
            if str(self.robomimic_root) not in sys.path:
                sys.path.insert(0, str(self.robomimic_root))
            environment = importlib.import_module("robomimic.envs.env_patcher_online")
            if not Path(environment.__file__).resolve().is_relative_to(self.robomimic_root):
                raise RuntimeError(
                    "A different robomimic installation is already imported; "
                    "restart ACQ4 with the configured custom fork first"
                )
            module = importlib.import_module(".model_runtime.policy", package=__package__)
            agent = getattr(module, _POLICY_CLASSES[phase])(model_path=checkpoint)
            agent.allow_goal_placeholders = False
            self._agents[phase] = agent
        return self._agents[phase]

    def run_inference(self, phase: str, observation: dict, *, goal=None) -> np.ndarray:
        with self._locks[phase]:
            agent = self._agent(phase)
            if agent.goal_required and goal is None:
                raise ValueError(f"{phase} policy requires an explicit goal")
            action = np.asarray(agent.inference(observation=observation, goal=goal, is_demo=False))
            if not np.isfinite(action).all():
                raise ValueError(f"{phase} policy returned a nonfinite action")
            return action

    def observation_input_width(self, phase: str, key: str, default_width: int = 15) -> int:
        with self._locks[phase]:
            agent = self._agent(phase)
            shapes = agent.importer.obs_shapes
            if key not in agent.get_required_obs_keys() and key not in shapes:
                return 0
            shape = shapes.get(key)
            return int(shape[-1]) if shape else default_width

    def start_attempt(self):
        """Discard recurrent state and frame histories from the previous attempt."""
        self._agents.clear()

    def provenance(self) -> dict:
        return {
            phase: {"checkpoint": str(path), "sha256": sha256(self.require(phase))}
            for phase, path in self.checkpoints.items()
        }
