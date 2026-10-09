"""ACQ4-facing PatcherBot attempt controls and service ownership."""

from __future__ import annotations

from acq4.modules.Module import Module
from acq4.util import Qt

from .models import PatcherBotDetector, PolicyRuntime
from .recording import AttemptRecorder


_PHASES = ("find pipette", "approach", "cell detect", "seal", "break in")
_MODES = ("Manual", "Classic", "Adaptive", "Agent", "Training")
_TERMINAL = ("out", "bath", "whole cell", "broken", "fouled")


class PatcherBotModule(Module):
    """Own the UI and services; ACQ4 owns the devices and state transitions.

    Config requires `patchPipette` and `camera` ACQ4 device names. Optional
    `detector`, `policy`, and `continuous` dictionaries describe artifacts and
    acquisition channels. Loading the module does not move or pressurize anything.
    """

    moduleDisplayName = "PatcherBot"
    moduleCategory = "Acquisition"

    def __init__(self, manager, name, config):
        super().__init__(manager, name, config)
        self.pipette = manager.getDevice(config["patchPipette"])
        from .patchpipette import PatcherBotPatchPipette

        if not isinstance(self.pipette, PatcherBotPatchPipette):
            raise TypeError(f"{config['patchPipette']} must be a PatcherBotPatchPipette device")
        self.camera = manager.getDevice(config["camera"])
        self.pipette.camera = self.camera
        self.pipette.continuous = None
        self.pipette.detector = None
        self.pipette.policy = None
        self.pipette.recorder = None
        self._recorder = None
        self._stream = None
        self._phase_seen = False
        detector = config.get("detector")
        if detector:
            self.pipette.detector = PatcherBotDetector(
                detector["pipetteCheckpoint"],
                cell_checkpoint=detector.get("cellCheckpoint"),
                cell_model=detector.get("cellModel", "segformer"),
                cell_encoder_config=detector.get("cellEncoderConfig"),
                device=detector.get("device"),
                input_range=detector.get("inputRange"),
            )
        policy = config.get("policy")
        if policy:
            self.pipette.policy = PolicyRuntime(
                policy.get("checkpoints", {}), robomimic_root=policy["robomimicRoot"]
            )
        self.pipette.sigStateChanged.connect(self._stateChanged)
        self.win = PatcherBotWindow(self)
        self.win.show()

    def prepareAttempt(self, state: str, modes: dict[str, str]):
        """Attach services before the UI or Autopatch starts one ACQ4 state job."""
        if state not in _PHASES:
            raise ValueError(f"Unknown start state {state!r}")
        if self._recorder is not None:
            raise RuntimeError("An attempt is already running")
        current = self.pipette.getState()
        if current is not None and current.stateName not in ("out", "bath"):
            raise RuntimeError(f"Cannot replace active pipette state {current.stateName!r}")
        self.manager.globalHalt.check()
        self.pipette.stateManager().configureModes(modes)
        chosen_mode = modes[state]
        if chosen_mode == "Agent":
            if state in ("approach", "cell detect"):
                raise RuntimeError(f"{state} Agent mode has no defined PatcherBot movement policy")
            phase = {"find pipette": "find_pipette", "seal": "gigaseal", "break in": "break_in"}[state]
            if self.pipette.policy is None:
                raise FileNotFoundError(f"{state} Agent mode needs a configured policy checkpoint")
            self.pipette.policy.require(phase)
            if self.pipette.detector is None:
                raise FileNotFoundError(f"{state} Agent mode needs a PipetteDetector4 checkpoint")
            self.pipette.detector.require("find_pipette")
            if state in ("seal", "break in") and not self.config.get("continuous"):
                raise RuntimeError(f"{state} Agent mode requires continuous membrane acquisition")
        if chosen_mode == "Adaptive" and state in ("find pipette", "approach", "cell detect"):
            if self.pipette.detector is None:
                raise FileNotFoundError(f"{state} Adaptive mode needs a PipetteDetector4 checkpoint")
            self.pipette.detector.require(
                {"find pipette": "find_pipette", "approach": "approach_cell", "cell detect": "hunt"}[state]
            )
        model_paths = {}
        for name, value in self.config.get("detector", {}).items():
            if name.endswith("Checkpoint"):
                model_paths[name] = value
        model_paths.update(self.config.get("policy", {}).get("checkpoints", {}))
        self._phase_seen = False
        self._recorder = AttemptRecorder(
            self.manager, self.pipette, config=self.config, models=model_paths
        )
        self.pipette.recorder = self._recorder
        try:
            self.pipette.newPatchAttempt()
            if self.pipette.policy is not None:
                self.pipette.policy.start_attempt()
            if self.config.get("continuous"):
                from .acquisition.continuous import ContinuousMembraneTest

                cfg = dict(self.config["continuous"])
                dev = self.manager.getDevice(cfg.pop("daqDevice"))
                if self.pipette.clampDevice is not None:
                    self.pipette.clampDevice.enableTestPulse(False, block=True)
                self._stream = ContinuousMembraneTest(self.manager, dev, cfg)
                if state in ("seal", "break in"):
                    self._stream.set_cell_mode(True)
                self._stream.start()
                self.pipette.continuous = self._stream
            self.pipette.setActive(True)
        except BaseException:
            try:
                if self.pipette.active:
                    self.pipette.setState("out")
                else:
                    self.pipette.stateManager().abort()
            finally:
                self._finishAttempt()
            raise

    def startAttempt(self, state: str, modes: dict[str, str]):
        """Start one UI-driven attempt with the same services as Autopatch."""
        self.prepareAttempt(state, modes)
        try:
            self.pipette.setState(state)
        except BaseException:
            self.abortAttempt()
            raise

    def _stateChanged(self, pipette, state, old):
        if pipette is not self.pipette:
            return
        if state in _PHASES:
            self._phase_seen = True
        if hasattr(self, "win"):
            self.win.status.setText(f"{old or '—'} → {state}")
        if state in _TERMINAL and self._phase_seen and self._recorder is not None:
            self._finishAttempt()

    def operatorResult(self, success: bool):
        job = self.pipette.getState()
        if job is None or not hasattr(job, "operatorResult"):
            raise RuntimeError("Current state is not awaiting an operator decision")
        if self._recorder is not None:
            self._recorder.operator_result(job.stateName, "success" if success else "abort")
        job.operatorResult(success)

    def abortAttempt(self):
        try:
            if self.pipette.active:
                # Out must initialize while active: deactivating first races the
                # ACQ4 fallback worker and leaves the clamp initialization undone.
                self.pipette.setState("out")
            else:
                self.pipette.stateManager().abort()
            if self.pipette.pressureDevice is not None:
                self.pipette.pressureDevice.setPressure(source="atmosphere", pressure=0)
        finally:
            self._finishAttempt()

    def finishAttempt(self):
        """Release logs and NI acquisition without altering a settled cell state."""
        self._finishAttempt()

    def _finishAttempt(self):
        try:
            if self._stream is not None:
                self._stream.stop()
        finally:
            self._stream = None
            self.pipette.continuous = None
            self._phase_seen = False
            if self._recorder is not None:
                try:
                    self._recorder.close()
                finally:
                    self._recorder = None
                    self.pipette.recorder = None

    def quit(self):
        if self._recorder is not None:
            self.abortAttempt()
        self.pipette.sigStateChanged.disconnect(self._stateChanged)
        self.win.close()
        super().quit()


class PatcherBotWindow(Qt.QWidget):
    """Minimal ACQ4 controls; never actuate a device without explicit arming."""

    def __init__(self, module: PatcherBotModule):
        super().__init__()
        self.module = module
        self.setWindowTitle("PatcherBot — ACQ4")
        layout = Qt.QVBoxLayout(self)
        self.status = Qt.QLabel("Idle")
        layout.addWidget(self.status)
        form = Qt.QFormLayout()
        self.startState = Qt.QComboBox()
        self.startState.addItems(_PHASES)
        form.addRow("Start phase", self.startState)
        self.modes = {}
        for state in _PHASES:
            combo = Qt.QComboBox()
            combo.addItems(_MODES)
            combo.setCurrentText("Manual")
            self.modes[state] = combo
            form.addRow(state, combo)
        layout.addLayout(form)
        self.armed = Qt.QCheckBox("Arm motion, pressure and amplifier commands")
        layout.addWidget(self.armed)
        start = Qt.QPushButton("Start attempt")
        start.clicked.connect(self._start)
        layout.addWidget(start)
        success = Qt.QPushButton("Operator: success")
        success.clicked.connect(lambda: self._operatorResult(True))
        layout.addWidget(success)
        failure = Qt.QPushButton("Operator: abort")
        failure.clicked.connect(lambda: self._operatorResult(False))
        layout.addWidget(failure)
        stop = Qt.QPushButton("Abort and vent")
        stop.clicked.connect(self._abort)
        layout.addWidget(stop)

    def _run(self, fn):
        try:
            fn()
        except Exception as exc:
            self.status.setText(f"Error: {type(exc).__name__}: {exc}")
            raise

    def _start(self):
        if not self.armed.isChecked():
            self.status.setText("Arm motion, pressure and amplifier commands before starting")
            return
        self.armed.setChecked(False)
        modes = {state: combo.currentText() for state, combo in self.modes.items()}
        self._run(lambda: self.module.startAttempt(self.startState.currentText(), modes))

    def _operatorResult(self, success: bool):
        self._run(lambda: self.module.operatorResult(success))

    def _abort(self):
        self._run(self.module.abortAttempt)
