"""Independent PatcherBot phase jobs for ACQ4's PatchPipette state manager.

All phase observations use ACQ4 SI hardware values at the boundary. The phase
algorithms below use PatcherBot's micrometres, millibar, and megaohms internally.
The absent robomimic checkpoints and detector checkpoints are never substituted:
Agent/Adaptive require explicit services; Training records but does not actuate.
"""
from __future__ import annotations

from collections import deque
from threading import Event
import time

import numpy as np

from acq4.devices.PatchPipette.states._base import PatchPipetteState
from acq4.motion import MoveSpec
from acq4.util.task import Stopped, check_stop, sleep

UM = 1e-6
MBAR = 100.0
MOHM = 1e6
MODES = ('Manual', 'Classic', 'Adaptive', 'Agent', 'Training')
# Training rows: one rig-recorder row per DAQ acquisition, polled every 42 ms.
TRAINING_ROW_PERIOD_S = 0.042


def frame_pixels(frame):
    """Row-major image array; ACQ4 Frame.data() is indexed [x, y]."""
    data = frame.data()
    return np.ascontiguousarray(np.swapaxes(data, 0, 1))


def resistance_features(history, input_width, slope_window):
    """Online form of SimpleDatasetBuilder's resistance_input/resistance_slope.

    *history* holds the current row last. resistance_input is the *prior* rows,
    oldest first, zero-padded on the left. resistance_slope is the mean of the
    last min(50, slope_window) row-to-row differences, 0 for rows 0-3; units
    are MΩ per row, not per second.
    """
    values = list(history)
    prior = values[:-1][-input_width:] if input_width else []
    features = {}
    if input_width:
        features['resistance_input'] = np.array(
            [0.0] * (input_width - len(prior)) + prior, dtype=np.float32)
    row = len(values) - 1
    slope = 0.0
    if row >= 4:
        n = min(min(50, max(1, int(slope_window))), row)
        slope = float(np.mean(np.diff(values[-(n + 1):])))
    features['resistance_slope'] = np.float32(slope)
    return features


def _number(value, label):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'Invalid {label}: {value!r}')
    return value

def _recordable(value):
    """Preserve sensor provenance and numerical arrays, never hand Qt objects to storage."""
    if hasattr(value, 'mapFromFrameToGlobal') and hasattr(value, 'info'):
        info = value.info()
        return {'frame_time': float(value.time), 'frame_id': _recordable(info.get('id'))}
    if hasattr(value, 'analysis') and hasattr(value, 'recording'):
        return {'test_pulse_started_at': float(value.recording.start_time),
                'test_pulse_analysis': _recordable(value.analysis)}
    if isinstance(value, dict):
        return {str(k): _recordable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_recordable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Cannot record unsupported phase value {type(value).__name__}')


class PhaseContext:
    """Per-job hardware adapter; no PatcherBot controller or global mutable state."""

    def __init__(self, job):
        self.job = job
        self.dev = job.dev
        self.pip = job.dev.pipetteDevice
        self.clamp = job.dev.clampDevice
        self.pressure = job.dev.pressureDevice
        self.camera = getattr(job.dev, 'camera', None)
        self.detector = getattr(job.dev, 'detector', None)
        self.policy = getattr(job.dev, 'policy', None)
        self.recorder = getattr(job.dev, 'recorder', None)
        self.last_frame_time = None
        self.last_observed_at = None
        # Policy rows of this job: enough for a 50-difference slope or the input window.
        self._resistance_history = deque(maxlen=51)


    def preflight(self):
        mode = self.job.config['mode']
        if mode not in MODES:
            raise ValueError(f'Unsupported {self.job.stateName} mode: {mode}')
        if mode == 'Agent':
            if self.job.stateName in ('approach', 'cell detect'):
                raise RuntimeError(f'{self.job.stateName} Agent has no defined PatcherBot movement action')
            if self.job.stateName == 'find pipette' and self.job.config['velocityPrediction']:
                raise RuntimeError('find pipette velocity policy needs a calibrated continuous velocity stage API')
            if self.policy is None:
                raise RuntimeError(f'{self.job.stateName} Agent needs dev.policy with a loaded checkpoint')
            self.policy.require(self.job.policy_phase)
            # Loads the checkpoint now: a broken policy must fail before any actuation.
            required = set(self.policy.required_obs_keys(self.job.policy_phase))
            self.policy.reset(self.job.policy_phase)  # fresh LSTM/chunk state per job
            if self.job.stateName in ('seal', 'break in') and getattr(self.dev, 'continuous', None) is None:
                raise RuntimeError(f'{self.job.stateName} Agent needs dev.continuous, not finite test pulses')
            if required & {'camera_image', 'pipette_positions'}:
                if self.detector is None or self.camera is None:
                    raise RuntimeError(f'{self.job.stateName} Agent needs dev.detector and dev.camera')
                self.detector.require('find_pipette')
        if mode == 'Adaptive' and self.job.stateName in ('find pipette', 'approach', 'cell detect'):
            if self.detector is None:
                raise RuntimeError(f'{self.job.stateName} Adaptive needs dev.detector with loaded checkpoints')
            self.detector.require(self.job.policy_phase)
        if self.job.stateName == 'find pipette' and mode != 'Manual':
            if self.detector is None:
                raise RuntimeError('find pipette needs dev.detector with loaded checkpoints')
            self.detector.require('find_pipette')
        if self.job.stateName in ('approach', 'cell detect') and self.pip.target is None and mode != 'Manual':
            raise RuntimeError(f'{self.job.stateName} requires an ACQ4 pipette target')
        if self.job.stateName in ('cell detect', 'seal', 'break in') and self.clamp is None and mode != 'Manual':
            raise RuntimeError(f'{self.job.stateName} requires an ACQ4 PatchClamp device')
        if self.job.stateName in ('cell detect', 'seal', 'break in') and self.pressure is None and mode not in ('Manual', 'Training'):
            raise RuntimeError(f'{self.job.stateName} requires an ACQ4 PressureControl device')
        if mode == 'Adaptive' and self.job.stateName in ('approach', 'cell detect'):
            if not callable(getattr(self.detector, 'detect_cells', None)):
                raise RuntimeError(f'Adaptive {self.job.stateName} needs dev.detector.detect_cells(image)')
            scope = getattr(self.camera, 'scopeDev', None)
            if scope is None or not callable(getattr(scope, 'setFocusDepth', None)):
                raise RuntimeError(f'Adaptive {self.job.stateName} needs calibrated camera.scopeDev focus control')
            if not self.detector.has_z_confidence:
                # PatcherBot's contact/spear gate needs Z confidence >= 0.5; it could never pass.
                raise RuntimeError(f'Adaptive {self.job.stateName} needs a PipetteDetector4 checkpoint '
                                   'with the Z-confidence head')
        if self.job.stateName in ('find pipette', 'approach', 'cell detect') and self.camera is None and mode != 'Manual':
            raise RuntimeError(f'{self.job.stateName} requires dev.camera')


    def monitor(self):
        if getattr(self.dev, 'continuous', None) is None:
            self.job.monitorTestPulse()


    def pulse(self, timeout=10):
        start = time.monotonic()
        continuous = getattr(self.dev, 'continuous', None)
        if continuous is not None:
            while True:
                check_stop()
                try:
                    sample = continuous.latest(max_age=2)
                except RuntimeError:
                    sample = None  # stream not yet ready or measurement too old
                if (sample is not None and self.job.stateName in ('seal', 'break in')
                        and sample.access_resistance_ohm is None):
                    sample = None  # pre-switch bath acquisition; wait for a fresh cell fit
                if sample is not None and sample.acquisition_id > getattr(self, 'last_acquisition_id', -1):
                    self.last_acquisition_id = sample.acquisition_id
                    timestamp = _number(sample.timestamp_monotonic_s, 'stream timestamp')
                    self.last_observed_at = time.monotonic()
                    return {'resistance': _number(sample.resistance_ohm, 'resistance') / MOHM,
                            'access_resistance': (np.nan if sample.access_resistance_ohm is None
                                                  else float(sample.access_resistance_ohm) / MOHM),
                            'membrane_resistance': (np.nan if sample.membrane_resistance_ohm is None
                                                    else float(sample.membrane_resistance_ohm) / MOHM),
                            'capacitance': (np.nan if sample.membrane_capacitance_farad is None
                                            else float(sample.membrane_capacitance_farad)),
                            'observed_at': timestamp, 'acquisition_id': sample.acquisition_id}
                if time.monotonic() - start >= timeout:
                    raise TimeoutError(f'{self.job.stateName}: no fresh continuous sample in {timeout}s')
                sleep(0.02)
        while True:
            check_stop()
            pulses = self.job.getTestPulses(timeout=0.2)
            if pulses:
                tp = pulses[-1]
                analysis = tp.analysis
                self.last_observed_at = time.monotonic()
                return {'resistance': _number(analysis['steady_state_resistance'], 'resistance') / MOHM,
                        'access_resistance': float(analysis.get('access_resistance', np.nan)) / MOHM,
                        'capacitance': float(analysis.get('capacitance', np.nan)),
                        'observed_at': time.monotonic(), 'test_pulse': tp}
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f'{self.job.stateName}: no fresh ACQ4 test pulse in {timeout}s')

    def window(self, size, interval):
        samples = []
        for index in range(size):
            if index:
                sleep(interval)
            samples.append(self.pulse())
        result = dict(samples[-1])
        result['resistance'] = float(np.mean([s['resistance'] for s in samples]))
        return result

    def image(self):
        with self.camera.ensureRunning():
            frame = self.camera.acquireFrames(1, ensureFreshFrames=True).wait()[0]
        timestamp = _number(frame.time, 'frame timestamp')
        if self.last_frame_time is not None and timestamp <= self.last_frame_time:
            raise RuntimeError('Stale camera frame; refusing motion')
        self.last_frame_time = timestamp
        self.last_observed_at = time.monotonic()
        return frame

    def detect_tip(self, frame):
        if self.detector is None:
            raise RuntimeError('Pipette detector checkpoint is unavailable')
        result = self.detector.detect_pipette_details(frame_pixels(frame))
        if result is None:
            raise RuntimeError('Pipette not detected in fresh frame')
        xy = np.asarray(result['xy'], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Invalid detector tip pixel coordinates')
        # z_confidence is None for checkpoints without the optional Z-confidence head.
        z_conf = result['z_confidence']
        return {'xy': xy, 'z_um': _number(result['z_um'], 'tip defocus'),
                'box_confidence': _number(result['box_confidence'], 'tip confidence'),
                'z_confidence': None if z_conf is None else _number(z_conf, 'focus confidence')}

    def image_delta_um(self, frame, dx, dy):
        """Map image displacement to global tip motion using the acquired frame calibration."""
        origin = np.asarray(frame.mapFromFrameToGlobal((0, 0, 0)), dtype=float)
        point = np.asarray(frame.mapFromFrameToGlobal((float(dx), float(dy), 0)), dtype=float)
        delta = (point - origin) / UM
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Invalid image-to-global calibration')
        return delta

    def position_um(self):
        return np.asarray(self.pip.globalPosition(), dtype=float) / UM

    def move_global(self, device, target, speed_um_s, *, clip=False):
        check_stop()
        if self.last_observed_at is None or time.monotonic() - self.last_observed_at > 2:
            raise RuntimeError('PatcherBot motion refused: observation is stale')
        current = np.asarray(device.globalPosition(), dtype=float)
        target = np.asarray(target, dtype=float)
        speed = _number(speed_um_s, 'motion speed')
        if current.shape != (3,) or target.shape != (3,) or not np.isfinite(target).all() or speed <= 0:
            raise ValueError('Unsafe phase global motion command')
        offset = target - current
        length = float(np.linalg.norm(offset))
        limit = _number(self.job.config['maxMoveStepUm'], 'maximum motion step') * UM
        if limit <= 0:
            raise ValueError('Maximum motion step must be positive')
        if length < 1e-12:
            return True
        arrived = length <= limit
        if not arrived:
            if not clip:
                raise RuntimeError(f'Planned {self.job.stateName} motion exceeds {limit / UM:.1f} µm step limit')
            target = current + offset * (limit / length)
        self.dev.dm.move(MoveSpec(device, target, speed=speed * UM),
                         name=f'PatcherBot {self.job.stateName} planned move').wait()
        check_stop()
        return arrived

    def move_um(self, offset, speed_um_s):
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Invalid phase relative motion command')
        return self.move_global(self.pip, np.asarray(self.pip.globalPosition(), dtype=float) + offset * UM,
                                speed_um_s, clip=True)

    def pressure_mbar(self, value):
        check_stop()
        if self.pressure is None:
            raise RuntimeError('PressureControl device is unavailable')
        self.pressure.setPressure(source='regulator', pressure=_number(value, 'pressure') * MBAR)

    def vent(self):
        if self.pressure is not None:
            self.pressure.setPressure(source='atmosphere', pressure=0)

    def infer(self, observation, goal=None):
        if self.policy is None:
            raise RuntimeError(f'{self.job.stateName} needs dev.policy with a loaded checkpoint')
        sample = {k: v for k, v in observation.items() if k not in ('frame', 'test_pulse')}
        frame = observation.get('frame')
        if frame is not None:
            sample['camera_image'] = self.detector.prepare_image(frame_pixels(frame))
        tip = observation.get('tip')
        if tip is not None:
            sample.update(pipette_image_xy=tip['xy'], pipette_defocus_um=tip['z_um'])
        position = observation.get('position_um')
        if position is not None:
            sample['manipulator_position'] = position
        if self.pressure is not None:
            sample['pressure'] = self.pressure.getPressure() / MBAR
            sample['commanded_pressure_mbar'] = self.pressure.pressure / MBAR
            sample['pressure_atm_state'] = self.pressure.getSource() == 'atmosphere'
        resistance = sample.get('resistance')
        if resistance is not None and np.isfinite(resistance):
            # One policy call is one training row.
            width = self.policy.observation_input_width(self.job.policy_phase, 'resistance_input')
            if not isinstance(width, int) or width < 0 or width > self._resistance_history.maxlen - 1:
                raise ValueError(f'Unsupported policy resistance_input width: {width!r}')
            self._resistance_history.append(float(resistance))
            sample.update(resistance_features(
                self._resistance_history, width, self.job.config.get('agentSlopeWindow', 20)))
        return self.policy.run_inference(self.job.policy_phase, sample, goal=goal)

    def record(self, observation, calculations, action):
        if self.recorder is not None:
            source = dict(observation)
            if frame := source.get('frame'):
                source['camera_image'] = frame_pixels(frame)
            details = {k: v for k, v in calculations.items() if k != 'observation'}
            self.recorder.record(self.job.stateName, _recordable(source),
                                 _recordable(details), _recordable(action))


class PatcherBotPhaseState(PatchPipetteState):
    """Observe -> calculate -> decide -> act, with ACQ4 cancellation and venting."""

    _parameterTreeConfig = {
        'mode': {'type': 'list', 'default': 'Classic', 'limits': list(MODES)},
        'maxMoveStepUm': {'type': 'float', 'default': 100., 'suffix': 'µm'},
    }
    _parameterDefaultOverrides = {'fallbackState': 'out'}
    policy_phase = None

    def __init__(self, dev, config=None):
        super().__init__(dev, config)
        self._operator_label = Event()
        self._operator_success = None

    def operatorResult(self, success: bool):
        """Publish a Training label; the worker handles it on its next cycle."""
        if self.config['mode'] != 'Training':
            raise RuntimeError('operatorResult only applies to Training mode')
        if not isinstance(success, bool):
            raise TypeError('operatorResult needs a Boolean label')
        self._operator_success = success
        self._operator_label.set()


    def runJob(self):
        # ACQ4 initializes pressure and finite test pulses before run(): reject
        # missing checkpoints/stream first and keep finite pulses off with NI stream.
        self.ctx = PhaseContext(self)
        try:
            self.ctx.preflight()
            if getattr(self.dev, 'continuous', None) is not None:
                self.config['initialTestPulseEnable'] = False
            try:
                return super().runJob()
            finally:
                try:
                    self.dev.pipetteDevice.stop()
                finally:
                    self.ctx.vent()
        except Stopped:
            raise
        except Exception as exc:
            try:
                self.ctx.record({}, {'error_type': type(exc).__name__, 'error': str(exc)},
                                {'kind': 'failed'})
            except Exception:
                self.logger.exception('Could not record PatcherBot phase failure')
            raise

    def _cleanup(self):
        try:
            self.dev.pipetteDevice.stop()
            if self.dev.pressureDevice is not None:
                self.dev.pressureDevice.setPressure(source='atmosphere', pressure=0)
        finally:
            super()._cleanup()

    def manual(self):
        self.setState('manual: awaiting operator state change')
        while True:
            sleep(0.1)

    def cycle(self, observation, state):
        if self.config['mode'] == 'Training' and self._operator_label.is_set():
            label = {'kind': 'complete' if self._operator_success else 'operator-failure'}
            self.ctx.record(observation, {}, label)
            if self._operator_success:
                return label
            raise RuntimeError(f'{self.stateName}: operator labelled this training attempt unsuccessful')
        calculations = self.calculate(observation, state)
        action = self.decide(calculations, state)
        self.ctx.record(observation, calculations, action)
        self.act(action, state)
        return action


class FindPipetteState(PatcherBotPhaseState):
    """Locate the tip; original 2 µm RMS pixel/Z goal and policy velocity/displacement semantics."""
    stateName = 'find pipette'
    policy_phase = 'find_pipette'
    _parameterTreeConfig = {**PatcherBotPhaseState._parameterTreeConfig,
        'toleranceUm': {'type': 'float', 'default': 2.0, 'suffix': 'µm'},
        'speedUmS': {'type': 'float', 'default': 200.0, 'suffix': 'µm/s'},
        'velocityPrediction': {'type': 'bool', 'default': False},
        'tipConfidence': {'type': 'float', 'default': 0.5},
    }

    def run(self):
        if self.config['mode'] == 'Manual':
            return self.manual()
        electrical_keys = {'resistance', 'resistance_input', 'resistance_slope'}
        needs_electrical = self.config['mode'] == 'Agent' and bool(
            electrical_keys & set(self.ctx.policy.required_obs_keys(self.policy_phase)))
        while True:
            check_stop()
            electrical = self.ctx.pulse() if needs_electrical else {}
            frame = self.ctx.image()
            try:
                tip = self.ctx.detect_tip(frame)
            except RuntimeError:
                sleep(0.05)
                continue
            obs = {'frame': frame, 'tip': tip, 'position_um': self.ctx.position_um(), **electrical}
            action = self.cycle(obs, None)
            if action['kind'] == 'complete':
                return {'state': 'approach'}

    def calculate(self, obs, state):
        frame, tip = obs['frame'], obs['tip']
        width, height = frame.data().shape[:2]  # ACQ4 frames are [x, y]
        target = np.array([width / 2, height / 2])
        error = target - tip['xy']
        delta = self.ctx.image_delta_um(frame, *error)
        delta[2] = -tip['z_um']
        goal_error = float(np.sqrt(np.mean(delta ** 2)))
        return {'error_um': delta, 'goal_error_um': goal_error, 'tip': tip, 'frame': frame,
                'observation': obs, 'target_px': target}

    def decide(self, calc, state):
        if calc['tip']['box_confidence'] < self.config['tipConfidence']:
            return {'kind': 'wait'}
        if calc['goal_error_um'] <= self.config['toleranceUm']:
            return {'kind': 'complete'}
        if self.config['mode'] == 'Training':
            return {'kind': 'wait'}
        if self.config['mode'] != 'Agent':
            return {'kind': 'move', 'delta_um': calc['error_um']}
        # Live PatcherBot goal: native-pixel tip position with zero defocus.
        goal = np.array([*calc['target_px'], 0.0])
        values = np.asarray(self.ctx.infer(calc['observation'], goal=goal), dtype=object).reshape(-1)
        if values.size not in (2, 3):
            raise ValueError('find_pipette policy needs 2 or 3 coordinates')
        xy = np.asarray(values[:2], dtype=float)
        z = (0.0 if self.config['velocityPrediction'] else calc['tip']['z_um']) if values.size == 2 or values[2] is None else float(values[2])
        if not np.isfinite(xy).all() or not np.isfinite(z):
            raise ValueError('Nonfinite find_pipette policy coordinates')
        if not self.config['velocityPrediction']:
            point = calc['tip']['xy'] + xy
            shape = calc['frame'].data().shape
            if np.any(point < 0) or np.any(point >= shape[:2]):
                return {'kind': 'wait'}
        delta = self.ctx.image_delta_um(calc['frame'], *xy)
        delta[2] = z
        return {'kind': 'move', 'delta_um': delta}

    def act(self, action, state):
        if action['kind'] == 'move':
            self.ctx.move_um(action['delta_um'], self.config['speedUmS'])
        elif action['kind'] == 'wait':
            sleep(0.05)


class ApproachCellState(PatcherBotPhaseState):
    """Plate direct alignment or slice two-step clearing to the target cell."""
    stateName = 'approach'
    policy_phase = 'approach_cell'
    _parameterTreeConfig = {**PatcherBotPhaseState._parameterTreeConfig,
        'cellType': {'type': 'list', 'default': 'Plate', 'limits': ['Plate', 'Slice']},
        'cellTypeToggle': {'type': 'bool', 'default': False},
        'cellDistanceUm': {'type': 'float', 'default': 20.0, 'suffix': 'µm'},
        'sliceStartDistanceUm': {'type': 'float', 'default': 75.0, 'suffix': 'µm'},
        'maxLocateSpeedUmS': {'type': 'float', 'default': 1000.0, 'suffix': 'µm/s'},
        'maxClearingSpeedUmS': {'type': 'float', 'default': 100.0, 'suffix': 'µm/s'},
        'tipConfidence': {'type': 'float', 'default': 0.5},
        'cellConfidence': {'type': 'float', 'default': 0.5},
        'maxShiftPx': {'type': 'float', 'default': 256.0, 'suffix': 'px'},
        'approachPipetteSpeedUmS': {'type': 'float', 'default': 20.0, 'suffix': 'µm/s'},
        'maxFocusStepUm': {'type': 'float', 'default': 5.0, 'suffix': 'µm'},
    }

    def run(self):
        if self.config['mode'] == 'Manual':
            return self.manual()
        if self.config['mode'] == 'Adaptive':
            return self._run_adaptive()
        target = np.asarray(self.ctx.pip.targetPosition(), dtype=float) / UM
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Approach target must be calibrated 3D global coordinates')
        state = {'target_um': target, 'step': 0}
        while True:
            check_stop()
            frame = self.ctx.image()
            obs = {'frame': frame, 'position_um': self.ctx.position_um(), 'target_um': target}
            if self.config['mode'] == 'Adaptive':
                obs['tip'] = self.ctx.detect_tip(frame)
            action = self.cycle(obs, state)
            if action['kind'] == 'complete':
                return {'state': 'cell detect'}

    def _run_adaptive(self):
        """Original search -> cell shift -> tip scan -> spear -> search cycle.

        The ACQ4 pipette target is a calibrated global cell location; the camera
        frame maps detector pixels into that same coordinate system.
        """
        target = np.asarray(self.ctx.pip.targetPosition(), dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Adaptive approach needs a calibrated 3D cell target')
        scope = self.ctx.camera.scopeDev
        focus = _number(scope.getFocusDepth(), 'scope focus')
        goal = self.config['cellDistanceUm'] if self.config['cellType'] == 'Plate' or self.config['cellTypeToggle'] else self.config['sliceStartDistanceUm']
        state = {'step': 0, 'cell_z': target[2], 'tip_z': focus,
                 'goal_um': goal, 'speared': False, 'spear_start': None}
        while True:
            check_stop()
            frame = self.ctx.image()
            focus = _number(scope.getFocusDepth(), 'scope focus')
            low, high = sorted((state['cell_z'], state['tip_z']))
            if not (low - 5 * UM <= focus <= high + 5 * UM):
                raise RuntimeError('Adaptive approach exceeded camera focus search bounds')
            tip = self.ctx.detect_tip(frame)
            cells = self.ctx.detector.detect_cells(frame_pixels(frame))
            obs = {'frame': frame, 'focus': focus, 'tip': tip, 'cells': cells,
                   'position_um': self.ctx.position_um(), 'target_um': target / UM}
            calculations = self.calculate_adaptive(obs, state)
            action = self.decide_adaptive(calculations, state)
            self.ctx.record(obs, calculations, action)
            self.act_adaptive(action, state)
            if action['kind'] == 'complete':
                return {'state': 'cell detect'}

    def calculate_adaptive(self, obs, state):
        frame = obs['frame']
        expected = np.asarray(frame.mapFromGlobalToFrame(self.ctx.pip.targetPosition()), dtype=float)[:2]
        cells = [np.asarray(cell, dtype=float) for cell in obs['cells']]
        cells = [cell for cell in cells if cell.shape == (3,) and np.isfinite(cell).all()
                 and cell[2] >= self.config['cellConfidence']]
        detected = min(cells, key=lambda cell: np.linalg.norm(cell[:2] - expected), default=None)
        cell_shift = None if detected is None else np.linalg.norm(detected[:2] - expected)
        tip = obs['tip']
        return {'observation': obs, 'cell': detected, 'cell_shift_px': cell_shift,
                'tip_ready': tip['box_confidence'] >= self.config['tipConfidence']
                and (tip['z_confidence'] or 0) >= 0.5,
                'focus_z_error_um': abs(obs['focus'] - state['tip_z']) / UM,
                'tip_defocus_um': abs(tip['z_um'])}

    def decide_adaptive(self, calc, state):
        obs = calc['observation']
        step = state['step']
        if step == 0:
            return {'kind': 'focus', 'target_z': state['cell_z'], 'next_step': 1}
        if step == 1:
            if calc['cell'] is None or calc['cell_shift_px'] > self.config['maxShiftPx']:
                return {'kind': 'wait'}
            return {'kind': 'cell_shift', 'cell': calc['cell'], 'frame': obs['frame']}
        if step == 2:
            if not calc['tip_ready']:
                return {'kind': 'wait'}
            if calc['focus_z_error_um'] <= 5 and calc['tip_defocus_um'] <= 2:
                return {'kind': 'pipette_shift', 'frame': obs['frame'], 'tip': obs['tip']}
            direction = -np.sign(obs['tip']['z_um'])
            return {'kind': 'focus', 'target_z': obs['focus'] + direction * self.config['maxFocusStepUm'] * UM}
        if step == 3:
            if state['speared']:
                return {'kind': 'complete'}
            depth = state['cell_z'] + state['goal_um'] * UM
            if depth >= self.ctx.pip.globalPosition()[2]:
                raise RuntimeError('Adaptive approach spear must advance toward the target')
            return {'kind': 'spear', 'target_depth': depth}
        if step == 4:
            if self.ctx.pip.globalPosition()[2] > state['spear_start'] + 0.5 * UM:
                raise RuntimeError('Adaptive approach spear moved in the wrong direction')
            return {'kind': 'focus', 'target_z': state['cell_z'], 'next_step': 1,
                    'after_spear': True}
        raise RuntimeError(f'Invalid adaptive approach step {step}')

    def act_adaptive(self, action, state):
        kind = action['kind']
        if kind not in ('wait', 'complete'):
            check_stop()
            if self.ctx.last_observed_at is None or time.monotonic() - self.ctx.last_observed_at > 2:
                raise RuntimeError('Adaptive motion refused: camera observation is stale')
        if kind == 'focus':
            scope = self.ctx.camera.scopeDev
            current = _number(scope.getFocusDepth(), 'scope focus')
            target = _number(action['target_z'], 'focus target')
            delta_um = np.clip((target - current) / UM,
                               -self.config['maxFocusStepUm'], self.config['maxFocusStepUm'])
            low, high = sorted((state['cell_z'], state['tip_z']))
            if not low - 5 * UM <= current + delta_um * UM <= high + 5 * UM:
                raise RuntimeError('Adaptive approach focus command exceeds search bounds')
            if abs(target - current) <= 0.5 * UM:
                if action.get('after_spear'):
                    state['speared'] = True
                if action.get('next_step') is not None:
                    state['step'] = action['next_step']
            else:
                check_stop()
                scope.setFocusDepth(current + delta_um * UM,
                                    speed=self.config['maxLocateSpeedUmS'] * UM).wait()
        elif kind == 'cell_shift':
            cell = action['cell']
            frame = action['frame']
            position = np.asarray(frame.mapFromFrameToGlobal((cell[0], cell[1], 0)), dtype=float)
            if not np.isfinite(position).all():
                raise ValueError('Invalid adaptive cell global location')
            current = np.asarray(self.ctx.camera.scopeDev.globalPosition(), dtype=float)
            target = current.copy()
            target[:2] = position[:2]
            arrived = self.ctx.move_global(
                self.ctx.camera.scopeDev, target, self.config['maxLocateSpeedUmS'], clip=True)
            if arrived:
                state['cell_z'] = _number(self.ctx.camera.scopeDev.getFocusDepth(), 'cell focus')
                state['step'] = 2
        elif kind == 'pipette_shift':
            tip = action['tip']
            frame = action['frame']
            position = np.asarray(frame.mapFromFrameToGlobal((tip['xy'][0], tip['xy'][1], 0)), dtype=float)
            target = np.asarray(self.ctx.pip.globalPosition(), dtype=float)
            target[:2] -= position[:2] - np.asarray(self.ctx.camera.scopeDev.globalPosition(), dtype=float)[:2]
            if not np.isfinite(target).all():
                raise ValueError('Invalid adaptive pipette centering')
            arrived = self.ctx.move_global(
                self.ctx.pip, target, self.config['approachPipetteSpeedUmS'], clip=True)
            if arrived:
                state['tip_z'] = _number(self.ctx.camera.scopeDev.getFocusDepth(), 'pipette focus')
                state['step'] = 3
        elif kind == 'spear':
            state['spear_start'] = self.ctx.pip.globalPosition()[2]
            pos = self.ctx.pip.positionAtDepth(action['target_depth'])
            if not np.isfinite(pos).all():
                raise ValueError('Invalid adaptive spear geometry')
            self.ctx.move_global(self.ctx.pip, pos, self.config['maxClearingSpeedUmS'])
            state['step'] = 4
        elif kind == 'wait':
            sleep(0.1)

    def calculate(self, obs, state):
        distance = (self.config['cellDistanceUm'] if self.config['cellType'] == 'Plate' or self.config['cellTypeToggle']
                    else self.config['sliceStartDistanceUm'])
        aim = obs['target_um'].copy()
        aim[2] += distance if state['step'] == 0 else self.config['cellDistanceUm']
        error = aim - obs['position_um']
        return {'observation': obs, 'error_um': error, 'step': state['step'], 'distance_um': distance}

    def decide(self, calc, state):
        if self.config['mode'] == 'Adaptive' and calc['observation']['tip']['box_confidence'] < self.config['tipConfidence']:
            return {'kind': 'wait'}
        if np.linalg.norm(calc['error_um']) <= 0.5:
            if self.config['cellType'] == 'Slice' and not self.config['cellTypeToggle'] and state['step'] == 0:
                return {'kind': 'next_step'}
            return {'kind': 'complete'}
        if self.config['mode'] == 'Training':
            return {'kind': 'wait'}
        return {'kind': 'move', 'error_um': calc['error_um'], 'step': calc['step']}

    def act(self, action, state):
        if action['kind'] == 'move':
            if state['step'] != action['step']:
                raise RuntimeError('Approach step changed since observation')
            self.ctx.move_um(action['error_um'], self.config['maxLocateSpeedUmS'] if state['step'] == 0
                             else self.config['maxClearingSpeedUmS'])
        elif action['kind'] == 'next_step':
            state['step'] += 1
        elif action['kind'] == 'wait':
            sleep(0.1)


class HuntCellState(PatcherBotPhaseState):
    """Hunt by five-reading resistance rise; Adaptive confirms image/depth contact."""
    stateName = 'cell detect'
    policy_phase = 'hunt'
    CONTACT_READINGS = 5
    _parameterTreeConfig = {**ApproachCellState._parameterTreeConfig,
        'pressureNearMbar': {'type': 'float', 'default': 45.0, 'suffix': 'mbar'},
        'cellRiseMohm': {'type': 'float', 'default': 0.300, 'suffix': 'MΩ'},
        'maxDistanceUm': {'type': 'float', 'default': 30.0, 'suffix': 'µm'},
        'descentSpeedUmS': {'type': 'float', 'default': 10.0, 'suffix': 'µm/s'},
        'stepUm': {'type': 'float', 'default': 0.4, 'suffix': 'µm'},
        'contactDistanceUm': {'type': 'float', 'default': 5.0, 'suffix': 'µm'},
        'tipConfidence': {'type': 'float', 'default': 0.5},
        'cellConfidence': {'type': 'float', 'default': 0.5},
        'limitTargetRadius': {'type': 'bool', 'default': True},
        'targetRadiusUm': {'type': 'float', 'default': 50., 'suffix': 'µm'},
    }
    _parameterDefaultOverrides = {**PatcherBotPhaseState._parameterDefaultOverrides,
                                  'initialTestPulseEnable': True, 'initialClampMode': 'VC'}

    def run(self):
        if self.config['mode'] == 'Manual':
            return self.manual()
        self.ctx.monitor()
        state = {'start': self.ctx.position_um(), 'baseline': None, 'readings': deque(maxlen=5),
                 'confirming': False, 'confirmed': 0, 'started': time.monotonic()}
        if self.config['mode'] == 'Adaptive':
            target = self.ctx.pip.targetPosition()
            scope = self.ctx.camera.scopeDev
            state['adaptive'] = {'step': 0, 'cell_z': target[2],
                                 'tip_z': scope.getFocusDepth(), 'speared': False,
                                 'goal_um': (self.config['cellDistanceUm'] if self.config['cellType'] == 'Plate'
                                             or self.config['cellTypeToggle'] else self.config['sliceStartDistanceUm']),
                                 'spear_start': None}
        if self.config['mode'] != 'Training':
            self.ctx.pressure_mbar(self.config['pressureNearMbar'])
            sleep(3)
        state['baseline'] = self.ctx.window(5, 0.001)['resistance']
        while True:
            obs = self.ctx.pulse()
            obs['position_um'] = self.ctx.position_um()
            if self.config['mode'] == 'Adaptive':
                frame = self.ctx.image()
                obs['frame'] = frame
                obs['tip'] = self.ctx.detect_tip(frame)
                obs['cells'] = self.ctx.detector.detect_cells(frame_pixels(frame))
            action = self.cycle(obs, state)
            if action['kind'] == 'complete':
                return {'state': 'seal'}

    def calculate(self, obs, state):
        if state['baseline'] is None:
            state['baseline'] = obs['resistance']
        state['readings'].append(obs['resistance'])
        travel = abs(float(obs['position_um'][2] - state['start'][2]))
        rising = len(state['readings']) >= 5 and obs['resistance'] - state['baseline'] >= self.config['cellRiseMohm']
        calc = {'observation': obs, 'travel_um': travel, 'resistance_delta_mohm': obs['resistance'] - state['baseline'],
                'resistance_up': rising}
        if self.config['mode'] == 'Adaptive':
            if time.monotonic() - state['started'] >= 60:
                raise TimeoutError('Adaptive hunt exceeded 60 second deadline')
            frame = obs['frame']
            tip = obs['tip']
            if tip['box_confidence'] < self.config['tipConfidence'] or (tip['z_confidence'] or 0) < 0.5:
                raise RuntimeError('Adaptive hunt lost fresh confident tip/depth detection')
            target = np.asarray(frame.mapFromGlobalToFrame(self.ctx.pip.targetPosition()), dtype=float)[:2]
            cells = [np.asarray(cell, dtype=float) for cell in obs['cells']]
            cells = [cell for cell in cells if cell.shape == (3,) and np.isfinite(cell).all()
                     and cell[2] >= self.config['cellConfidence']]
            cell = min(cells, key=lambda item: np.linalg.norm(item[:2] - target), default=None)
            if cell is not None and self.config['limitTargetRadius']:
                offset = self.ctx.image_delta_um(frame, *(cell[:2] - target))
                if np.linalg.norm(offset) > self.config['targetRadiusUm']:
                    cell = None
            aligned = (cell is not None and np.linalg.norm(self.ctx.image_delta_um(frame, *(tip['xy'] - cell[:2])))
                       <= self.config['contactDistanceUm'])
            scope_focus = self.ctx.camera.scopeDev.getFocusDepth()
            touching = aligned and abs(tip['z_um']) <= 2 and abs(scope_focus - state['adaptive']['cell_z']) <= 2 * UM
            if state['confirming']:
                state['confirmed'] = state['confirmed'] + 1 if rising and touching else 0
            calc.update(touching=touching, aligned=aligned, confirmed=state['confirmed'])
        return calc

    def decide(self, calc, state):
        mode = self.config['mode']
        if calc['travel_um'] >= self.config['maxDistanceUm'] and mode != 'Training':
            raise RuntimeError('Cell not detected before max hunt distance')
        if mode == 'Adaptive':
            if calc['confirmed'] >= self.CONTACT_READINGS:
                return {'kind': 'complete'}
            if calc['resistance_up'] or calc['touching']:
                return {'kind': 'confirm'}
            if state['confirming']:
                if time.monotonic() - state['confirmation_started'] >= 5:
                    raise TimeoutError('Adaptive hunt contact not confirmed in five seconds')
                return {'kind': 'wait'}
            move_calc = ApproachCellState.calculate_adaptive(self, calc['observation'], state['adaptive'])
            motion = ApproachCellState.decide_adaptive(self, move_calc, state['adaptive'])
            if motion['kind'] == 'complete':
                return {'kind': 'confirm'}
            if motion['kind'] == 'spear' and not calc['aligned']:
                raise RuntimeError('Adaptive hunt lost cell/tip alignment during spear')
            return {'kind': 'adaptive', 'motion': motion}
        if mode in ('Training', 'Manual'):
            return {'kind': 'wait'}
        if mode == 'Agent':
            raise RuntimeError('Source Hunt Agent defines no movement policy action; use Classic or Adaptive')
        return {'kind': 'move', 'distance_um': self.config['stepUm']}

    def act(self, action, state):
        if action['kind'] == 'move':
            distance = _number(action['distance_um'], 'hunt step')
            if distance <= 0 or distance > self.config['maxDistanceUm']:
                raise ValueError('Hunt step out of bounds')
            self.ctx.move_um((0, 0, -distance), self.config['descentSpeedUmS'])
        elif action['kind'] == 'confirm' and not state['confirming']:
            self.ctx.pip.stop()
            state.update(confirming=True, confirmed=0, confirmation_started=time.monotonic())
        elif action['kind'] == 'adaptive':
            ApproachCellState.act_adaptive(self, action['motion'], state['adaptive'])
        elif action['kind'] == 'wait':
            sleep(0.04)


class GigasealState(PatcherBotPhaseState):
    """Five-sample seal slope pressure ramp with progress deadline and 3 successes."""
    stateName = 'seal'
    policy_phase = 'gigaseal'
    _parameterTreeConfig = {**PatcherBotPhaseState._parameterTreeConfig,
        'targetMohm': {'type': 'float', 'default': 1000.0, 'suffix': 'MΩ'},
        'minDeltaMohm': {'type': 'float', 'default': 15.0, 'suffix': 'MΩ'},
        'holdSwitch': {'type': 'float', 'default': 12.0},
        'increaseSlopeGate': {'type': 'float', 'default': 3000.0},
        'constantSlopeGate': {'type': 'float', 'default': 10.0},
        'decreaseSlopeGate': {'type': 'float', 'default': 5.0},
        'measurementIntervalS': {'type': 'float', 'default': 0.2, 'suffix': 's'},
        'deadlineS': {'type': 'float', 'default': 150.0, 'suffix': 's'},
        'rampLimitMbar': {'type': 'float', 'default': -30.0, 'suffix': 'mbar'},
        'holdingV': {'type': 'float', 'default': -0.07, 'suffix': 'V'},
        # Agent: one policy call per training row (rig-recorder rows, ~42 ms apart).
        'agentRowPeriodS': {'type': 'float', 'default': TRAINING_ROW_PERIOD_S, 'suffix': 's'},
        'agentSlopeWindow': {'type': 'int', 'default': 20},
        # Policy dim 0 is applied pressure in mbar, 0 meaning ATM in training data.
        'agentVentAboveMbar': {'type': 'float', 'default': -2.5, 'suffix': 'mbar'},
        'agentPressureStepMbar': {'type': 'float', 'default': 5.0, 'suffix': 'mbar'},
    }
    _parameterDefaultOverrides = {**PatcherBotPhaseState._parameterDefaultOverrides,
                                  'initialClampMode': 'VC', 'initialTestPulseEnable': True}

    def run(self):
        if self.config['mode'] == 'Manual':
            return self.manual()
        if getattr(self.dev, 'continuous', None) is not None:
            self.dev.continuous.set_cell_mode(True)
        self.ctx.monitor()
        interval = self.config['measurementIntervalS']
        if interval <= 0:
            raise ValueError('Measurement interval must be positive')
        first = self.ctx.window(5, interval)
        state = {'previous': first['resistance'], 'pressure': -5.0, 'prev_pressure': -5.0,
                 'last_progress': time.monotonic(), 'success': 0, 'holding_switched': False,
                 'release_reference': None, 'bad_cell': 0, 'last_agent_action': None,
                 'observations_since_last_action': 0}
        self.ctx.vent()
        sleep(3)
        if self.config['mode'] in ('Classic', 'Adaptive'):
            self.ctx.pressure_mbar(-5)
        if self.config['mode'] == 'Agent':
            return self._run_agent(state)
        while True:
            obs = self.ctx.window(5, interval)
            action = self.cycle(obs, state)
            if action['kind'] == 'complete':
                return {'state': 'break in'}

    def _run_agent(self, state):
        period = self.config['agentRowPeriodS']
        if not period > 0:
            raise ValueError('agentRowPeriodS must be positive')
        now = time.monotonic()
        state.update(progress_reference=state['previous'], progress_reference_at=now,
                     above_target_since=None)
        next_row = now
        while True:
            check_stop()
            obs = self.ctx.pulse()  # one fresh acquisition is one policy row
            action = self.cycle(obs, state)
            if action['kind'] == 'complete':
                return {'state': 'break in'}
            next_row = max(next_row + period, time.monotonic())
            sleep(next_row - time.monotonic())

    def calculate(self, obs, state):
        resistance = obs['resistance']
        if self.config.get('mode') == 'Agent':
            # Progress over the Classic window span (5 samples), not per 42 ms row.
            now = time.monotonic()
            span = 5 * self.config['measurementIntervalS']
            delta = resistance - state['progress_reference']
            if now - state['progress_reference_at'] >= span:
                if delta >= self.config['minDeltaMohm']:
                    state['last_progress'] = now
                state.update(progress_reference=resistance, progress_reference_at=now)
            return {'observation': obs, 'resistance_mohm': resistance, 'delta_mohm': delta}
        delta = resistance - state['previous']
        if delta >= self.config['minDeltaMohm']:
            state['last_progress'] = time.monotonic()
        state['previous'] = resistance
        return {'observation': obs, 'resistance_mohm': resistance,
                'delta_mohm': delta, 'rate_mohm_s': delta / (5 * self.config['measurementIntervalS'])}

    def decide(self, calc, state):
        if time.monotonic() - state['last_progress'] >= self.config['deadlineS']:
            raise TimeoutError('Seal resistance failed to improve before deadline')
        resistance = calc['resistance_mohm']
        mode = self.config['mode']
        if mode == 'Agent':
            # Same sustained span as three consecutive 5-sample Classic windows.
            if resistance >= self.config['targetMohm']:
                since = state['above_target_since'] = state['above_target_since'] or time.monotonic()
                if time.monotonic() - since >= 15 * self.config['measurementIntervalS']:
                    return {'kind': 'complete'}
            else:
                state['above_target_since'] = None
        else:
            state['success'] = state['success'] + 1 if resistance >= self.config['targetMohm'] else 0
            if state['success'] >= 3:
                return {'kind': 'complete'}
        holding = resistance >= self.config['targetMohm'] / self.config['holdSwitch'] and not state['holding_switched']
        if mode in ('Training', 'Manual'):
            return {'kind': 'wait', 'holding': False}
        if state['release_reference'] is not None and mode in ('Classic', 'Adaptive'):
            difference = resistance - state['release_reference']
            if difference < 0:
                state['bad_cell'] += 1
                if state['bad_cell'] > 5:
                    raise RuntimeError('Bad cell after seal pressure release')
            return {'kind': 'reset', 'pressure_mbar': -5., 'holding': holding,
                    'release_difference_mohm': difference}
        if mode == 'Agent':
            raw = np.asarray(self.ctx.infer(calc['observation']), dtype=float).reshape(-1)
            if raw.size != 2 or not np.isfinite(raw).all():
                raise ValueError('gigaseal policy must return [applied_pressure_mbar, '
                                 f'observations_until_next_action], got {raw!r}')
            atm = bool(raw[0] > self.config['agentVentAboveMbar'])
            step = self.config['agentPressureStepMbar']
            # Never positive pressure on a seal; suction is limited to rampLimitMbar.
            pressure = state['pressure'] if atm else float(
                np.clip(np.round(raw[0] / step) * step, self.config['rampLimitMbar'], -5))
            return {'kind': 'pressure', 'pressure_mbar': pressure, 'atm': atm, 'holding': holding,
                    'policy_applied_pressure_mbar': float(raw[0]),
                    'policy_observations_until_next_action': float(raw[1])}
        rate = calc['rate_mohm_s']
        pressure = state['pressure']
        speed = 1
        if rate < self.config['targetMohm'] / self.config['increaseSlopeGate']:
            pressure -= 5
            speed = 3
        elif rate <= self.config['targetMohm'] / self.config['constantSlopeGate']:
            speed = 1
        elif rate <= self.config['targetMohm'] / self.config['decreaseSlopeGate']:
            pressure += 5
            speed = 3
        pressure = max(min(pressure, -5), self.config['rampLimitMbar'])
        return {'kind': 'pressure', 'pressure_mbar': pressure, 'atm': pressure <= self.config['rampLimitMbar'],
                'speed': speed, 'holding': holding}

    def act(self, action, state):
        if self.config['mode'] == 'Agent' and action['kind'] == 'pressure':
            check_stop()
            if self.ctx.last_observed_at is None or time.monotonic() - self.ctx.last_observed_at > 2:
                raise RuntimeError('Stale Agent seal observation; refusing pressure update')
            if action.get('holding'):
                self.ctx.clamp.setHolding(mode='VC', value=self.config['holdingV'])
                state['holding_switched'] = True
            applied = (action['pressure_mbar'], action['atm'])
            if applied == state['last_agent_action']:
                state['observations_since_last_action'] += 1
                return
            if action['atm']:
                # Keep the regulator setpoint; state cleanup winds it down to zero.
                self.ctx.pressure.setPressure(source='atmosphere')
            else:
                self.ctx.pressure_mbar(action['pressure_mbar'])
            state.update(last_agent_action=applied, observations_since_last_action=0,
                         pressure=action['pressure_mbar'])
            return
        if action.get('holding'):
            self.ctx.clamp.setHolding(mode='VC', value=self.config['holdingV'])
            state['holding_switched'] = True
        if action['kind'] == 'pressure':
            if action['atm']:
                self.ctx.vent()
                if state['release_reference'] is None and self.config['mode'] in ('Classic', 'Adaptive'):
                    state['release_reference'] = state['previous']
                    sleep(5)
            else:
                self.ctx.pressure_mbar(action['pressure_mbar'])
                state['release_reference'] = None
            state['pressure'] = action['pressure_mbar']
            if action['pressure_mbar'] != state['prev_pressure']:
                sleep(5 / action.get('speed', 1))
                state['prev_pressure'] = action['pressure_mbar']
        elif action['kind'] == 'reset':
            self.ctx.pressure_mbar(action['pressure_mbar'])
            state['pressure'] = action['pressure_mbar']
            state['release_reference'] = None
        else:
            sleep(0.1)


class BreakInState(PatcherBotPhaseState):
    """Three consecutive low-access readings; pressure pulse/zap trial schedule."""
    stateName = 'break in'
    policy_phase = 'break_in'
    _parameterTreeConfig = {**PatcherBotPhaseState._parameterTreeConfig,
        'maxAccessMohm': {'type': 'float', 'default': 70.0, 'suffix': 'MΩ'},
        'pulsePressureMbar': {'type': 'float', 'default': -345.0, 'suffix': 'mbar'},
        'pulsePressureDurationS': {'type': 'float', 'default': 1.0, 'suffix': 's'},
        'zap': {'type': 'bool', 'default': False},
        'zapDurationS': {'type': 'float', 'default': 25e-6, 'suffix': 's'},
    }
    _parameterDefaultOverrides = {**PatcherBotPhaseState._parameterDefaultOverrides,
                                  'initialPressureSource': 'atmosphere', 'initialClampMode': 'VC',
                                  'initialTestPulseEnable': True}

    def run(self):
        if self.config['mode'] == 'Manual':
            return self.manual()
        self.ctx.monitor()
        state = {'trials': 0, 'good': 0}
        while True:
            obs = self.ctx.window(5, 0.2)
            if self.config['mode'] == 'Agent':
                obs['frame'] = self.ctx.image()
                obs['tip'] = self.ctx.detect_tip(obs['frame'])
                obs['position_um'] = self.ctx.position_um()
            action = self.cycle(obs, state)
            if action['kind'] == 'complete':
                return {'state': 'whole cell'}

    def calculate(self, obs, state):
        trial = state['trials'] + 1
        return {'observation': obs, 'trial': trial, 'wait_s': 0.5 * (1 + trial / 2),
                'pulse_duration_s': 1 / (self.config['pulsePressureDurationS'] * (2 if trial >= 5 else 1))}

    def decide(self, calc, state):
        obs = calc['observation']
        access = _number(obs['access_resistance'], 'access resistance')
        state['good'] = state['good'] + 1 if access <= self.config['maxAccessMohm'] else 0
        if state['good'] >= 3:
            return {'kind': 'complete'}
        if access <= self.config['maxAccessMohm']:
            return {'kind': 'wait'}
        mode = self.config['mode']
        if mode in ('Training', 'Manual'):
            return {'kind': 'wait'}
        if mode == 'Agent':
            raw = np.asarray(self.ctx.infer({k: v for k, v in obs.items() if k not in ('access_resistance', 'capacitance', 'test_pulse')}), dtype=float).reshape(-1)
            if raw.size not in (1, 2) or not np.isfinite(raw).all():
                raise ValueError('Invalid break-in policy ATM/zap command')
            return {'kind': 'agent', 'atm': bool(raw[0] >= 0.5),
                    'zap': bool(raw[1] >= 0.5) if raw.size == 2 else False, 'trial': calc['trial'],
                    'wait_s': calc['wait_s']}
        return {'kind': 'pulse', 'trial': calc['trial'], 'wait_s': calc['wait_s'],
                'duration_s': calc['pulse_duration_s'], 'zap_count': 2 if calc['trial'] % 3 == 0 else 0}

    def act(self, action, state):
        if action['kind'] == 'wait':
            sleep(0.2)
            return
        if action['kind'] == 'complete':
            self.ctx.vent()
            return
        if action['kind'] == 'agent':
            check_stop()
            if self.ctx.last_observed_at is None or time.monotonic() - self.ctx.last_observed_at > 2:
                raise RuntimeError('Stale Agent break-in observation; refusing pressure/zap')
        state['trials'] = action['trial']
        if action['kind'] == 'pulse':
            try:
                self.ctx.pressure_mbar(self.config['pulsePressureMbar'])
                sleep(action['duration_s'])
            finally:
                self.ctx.vent()
            sleep(action['wait_s'])
            zaps = action['zap_count']
            delay = 1
        else:
            if action['atm']:
                self.ctx.vent()
            else:
                self.ctx.pressure_mbar(self.config['pulsePressureMbar'])
            zaps = int(action['zap'])
            delay = action['wait_s']
        for _ in range(zaps if self.config['zap'] else 0):
            check_stop()
            self.ctx.clamp.zap(duration=self.config['zapDurationS'])
            sleep(0.5)
        sleep(delay)
        if action['kind'] == 'pulse' and state['trials'] > 15:
            raise RuntimeError('Break-in failed after 15 trials')
