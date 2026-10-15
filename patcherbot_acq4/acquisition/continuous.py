"""Continuous NI membrane-test stream with ACQ4 device ownership.

The waveform, resistance split and mono-exponential equivalent-circuit calculation
are adapted from ``patcherbot/devices/amplifier/DAQ.py``. All public electrical
quantities use SI units; the DAQ voltage-to-cell gain is rig-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class MembraneSample:
    acquisition_id: int
    timestamp_monotonic_s: float
    time_s: np.ndarray
    read_v: np.ndarray
    response_a: np.ndarray
    raw: np.ndarray  # two AI channels in DAQ volts, readback then response
    resistance_ohm: float
    access_resistance_ohm: float | None
    membrane_resistance_ohm: float | None
    membrane_capacitance_farad: float | None


class ContinuousMembraneTest:
    """Own a NiDAQ while acquiring a regenerating voltage-clamp test pulse.

    ``pause`` stops and closes both NI tasks, releases the ACQ4 device lock and
    waits for the acquisition worker to park. ``resume`` reserves the device
    again and creates a freshly synchronized AI/AO pair. Never invoke a finite
    protocol against these channels before ``pause`` returns.

    Configuration channels may be ``/Dev1/ai0`` / ``Dev1/ai0``, or relative
    channels if the ACQ4 NiDAQ has an explicit ``device`` setting. The command
    amplitude is volts at the cell; gain factors are physical SI units per DAQ
    volt (the PatcherBot V-clamp defaults are 0.02 V/V and 2e-9 A/V).
    """

    def __init__(
        self,
        manager: Any,
        daq_device: Any,
        config: Mapping[str, Any],
        *,
        nidaqmx_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.manager = manager
        self.daq_device = daq_device
        self._factory = nidaqmx_factory or (lambda: importlib.import_module('nidaqmx'))
        self.sample_rate = float(config.get('sampleRate', 50_000))
        self.pulse_frequency = float(config.get('pulseFrequency', 40))
        self.command_amplitude = float(config.get('commandAmplitude', 0.010))
        self.command_gain = float(config.get('commandVoltsPerDaqVolt', 0.020))
        self.read_gain = float(config.get('readVoltsPerDaqVolt', 0.020))
        self.response_gain = float(config.get('responseAmpsPerDaqVolt', 2e-9))
        for name, val in (
            ('sampleRate', self.sample_rate), ('pulseFrequency', self.pulse_frequency),
            ('commandAmplitude', self.command_amplitude),
            ('commandVoltsPerDaqVolt', self.command_gain),
            ('readVoltsPerDaqVolt', self.read_gain),
            ('responseAmpsPerDaqVolt', self.response_gain),
        ):
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        periods = self.sample_rate / self.pulse_frequency
        if periods != int(periods) or periods < 32:
            raise ValueError('sampleRate / pulseFrequency must be an integer >= 32 for regeneration')
        self._period = int(periods)
        self._read_channel = self._channel(config['readChannel'], 'ai')
        self._response_channel = self._channel(config['responseChannel'], 'ai')
        self._command_channel = self._channel(config['commandChannel'], 'ao')
        if self._read_channel == self._response_channel:
            raise ValueError('readChannel and responseChannel must be distinct')
        physical_device = self._read_channel.split('/')[0]
        trigger = config.get('triggerTerminal')
        self._trigger = (
            f'/{physical_device.split("Mod")[0] if "Mod" in physical_device else physical_device}/ai/StartTrigger'
            if trigger is None else str(trigger)
        )
        if not self._trigger.startswith('/') or self._trigger == '/':
            raise ValueError('triggerTerminal must be an absolute NI terminal path')

        # Same 0.5 ms leading baseline and 50% duty cycle as PatcherBot's
        # createSquareWave(40, 50000, 0.5, 0.5, 0.025). Regeneration repeats
        # exactly one integer period, including the baseline padding.
        pad = int(0.0005 * self.sample_rate)
        high = self._period // 2
        if pad < 1 or pad + high >= self._period:
            raise ValueError('sampleRate/pulseFrequency cannot represent the test pulse')
        self._wave = np.zeros(self._period)
        self._wave[pad:pad + high] = self.command_amplitude / self.command_gain
        if np.max(self._wave) > 10:
            raise ValueError('commandAmplitude / commandVoltsPerDaqVolt exceeds 10 V AO range')
        self._time_axis = np.arange(self._period, dtype=float) / self.sample_rate
        self._time_axis.setflags(write=False)
        self._lifecycle_lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._run = threading.Event()
        self._stop = threading.Event()
        self._cell_mode = threading.Event()  # Bath: total R only; cell: fit Ra/Rm/Cm.
        self._mode_generation = 0
        self._ready = threading.Event()
        self._parked = threading.Event()
        self._worker: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._error: BaseException | None = None
        self._fatal_error: BaseException | None = None
        self._sample: MembraneSample | None = None
        self._acquisition_id = 0
        self._nidaqmx: Any = None
        self._ai: Any = None
        self._ao: Any = None
        self._ai_started = False
        self._ao_started = False

    def _channel(self, channel: str, kind: str) -> str:
        channel = str(channel).lstrip('/')
        if '/' not in channel:
            device = getattr(self.daq_device, 'exclusiveDevice', None)
            if not device:
                raise ValueError(f'{kind} channel {channel!r} needs a physical device prefix')
            channel = f'{device}/{channel}'
        device = getattr(self.daq_device, 'exclusiveDevice', None)
        if device and not channel.startswith(f'{device}/'):
            raise ValueError(f'{channel!r} does not belong to ACQ4 NiDAQ {device!r}')
        if not channel.split('/')[-1].startswith(kind):
            raise ValueError(f'{channel!r} must be an {kind} channel')
        return channel

    def start(self) -> None:
        """Start streaming; raise immediately on missing NI driver or setup errors."""
        with self._lifecycle_lock:
            if self._worker is not None:
                raise RuntimeError('Continuous membrane test is already started')
            self.manager.globalHalt.check()
            try:
                self._nidaqmx = self._factory()
            except ImportError as exc:
                raise RuntimeError('Continuous membrane test requires the nidaqmx Python package and NI-DAQmx driver') from exc
            self._stop.clear()
            self._ready.clear()
            self._parked.clear()
            self._run.set()
            self._startup_error = None
            self._fatal_error = None
            with self._sample_lock:
                self._error = None
                self._sample = None
            self.manager.globalHalt.add_abort_callback(self.stop, name='ContinuousMembraneTest.stop')
            self._worker = threading.Thread(target=self._acquire, name='ContinuousMembraneTest', daemon=True)
            try:
                self._worker.start()
            except BaseException:
                self._worker = None
                self.manager.globalHalt.remove_abort_callback(self.stop)
                raise
            self._ready.wait()
            if self._startup_error is not None:
                self._worker.join()
                self._worker = None
                self.manager.globalHalt.remove_abort_callback(self.stop)
                raise RuntimeError('Cannot start continuous NI membrane acquisition') from self._startup_error

    def stop(self) -> None:
        """Stop AO/AI, release the ACQ4 reservation, and invalidate old data."""
        with self._lifecycle_lock:
            worker = self._worker
            if worker is None:
                return
            self._stop.set()
            self._run.set()
            if worker is threading.current_thread():
                return
            worker.join()
            self._worker = None
            self.manager.globalHalt.remove_abort_callback(self.stop)
            with self._sample_lock:
                self._sample = None

    def pause(self, timeout: float = 2.0) -> None:
        """Block until tasks are closed and the device is free for finite protocols."""
        with self._lifecycle_lock:
            if self._worker is None or not self._worker.is_alive():
                raise RuntimeError('Continuous membrane test is not running')
            self._run.clear()
            if not self._parked.wait(timeout):
                raise TimeoutError('Continuous NI acquisition did not release the device in time')
            with self._sample_lock:
                self._sample = None
                if self._fatal_error is not None:
                    raise RuntimeError('Continuous NI acquisition failed while pausing') from self._fatal_error
            if not self._worker.is_alive():
                raise RuntimeError('Continuous NI acquisition stopped while pausing')

    def resume(self) -> None:
        """Reacquire the NI device and restart a synchronized stream."""
        with self._lifecycle_lock:
            if self._worker is None or not self._worker.is_alive():
                raise RuntimeError('Continuous membrane test is not running')
            self.manager.globalHalt.check()
            self._ready.clear()
            self._parked.clear()
            self._run.set()
            self._ready.wait()
            with self._sample_lock:
                if self._fatal_error is not None:
                    raise RuntimeError('Cannot resume continuous NI acquisition') from self._fatal_error

    def set_cell_mode(self, enabled: bool) -> None:
        """Switch between bath resistance and cell equivalent-circuit fit."""
        with self._sample_lock:
            if self._cell_mode.is_set() == bool(enabled):
                return
            if enabled:
                self._cell_mode.set()
            else:
                self._cell_mode.clear()
            self._mode_generation += 1
            self._sample = None
            self._error = None

    def latest(self, max_age: float | None = None) -> MembraneSample:
        """Return latest valid period; reject missing, failed or stale measurements."""
        if max_age is None:
            max_age = 2.5 / self.pulse_frequency
        if not math.isfinite(max_age) or max_age < 0:
            raise ValueError('max_age must be finite and nonnegative')
        with self._sample_lock:
            sample, error = self._sample, self._error
        if error is not None:
            raise RuntimeError('Continuous NI membrane test failed') from error
        if sample is None or not self._run.is_set() or self._stop.is_set():
            raise RuntimeError('No valid continuous membrane sample available')
        if time.monotonic() - sample.timestamp_monotonic_s > max_age:
            raise RuntimeError('Continuous membrane sample is stale')
        return sample

    def _acquire(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._run.is_set():
                    self._parked.set()
                    self._run.wait()
                    self._parked.clear()
                    if self._stop.is_set():
                        break
                self.manager.globalHalt.check()
                # DeviceLocker is thread-affine: the worker both acquires and
                # releases it, never the caller of start()/pause()/stop().
                with self.manager.reserveDevices([self.daq_device], reserver='ContinuousMembraneTest'):
                    self.manager.globalHalt.check()
                    if not self._run.is_set() or self._stop.is_set():
                        continue
                    try:
                        self._open_tasks()
                        with self._sample_lock:
                            self._sample = None
                            self._error = None
                        self._ready.set()
                        while self._run.is_set() and not self._stop.is_set():
                            raw = np.asarray(
                                self._ai.read(number_of_samples_per_channel=self._period, timeout=1.0),
                                dtype=float,
                            )
                            now = time.monotonic()
                            if raw.shape != (2, self._period) or not np.isfinite(raw).all():
                                raise ValueError(f'Invalid NI input buffer shape or values: {raw.shape}')
                            with self._sample_lock:
                                generation = self._mode_generation
                                cell_mode = self._cell_mode.is_set()
                            raw.setflags(write=False)
                            try:
                                sample = self._process(raw, cell_mode=cell_mode)
                            except ValueError as exc:
                                with self._sample_lock:
                                    if generation == self._mode_generation:
                                        self._sample = None
                                        self._error = exc
                            else:
                                with self._sample_lock:
                                    if generation == self._mode_generation:
                                        self._acquisition_id += 1
                                        self._sample = MembraneSample(self._acquisition_id, now, *sample)
                                        self._error = None
                            self._stop.wait(1 / self.pulse_frequency)
                    finally:
                        self._close_tasks()
                self._ready.clear()
                self._parked.set()
        except BaseException as exc:
            with self._sample_lock:
                self._sample = None
                self._error = exc
                self._fatal_error = exc
            if not self._ready.is_set():
                self._startup_error = exc
            self._stop.set()
            self._run.set()
        finally:
            self._ready.set()
            self._parked.set()

    def _open_tasks(self) -> None:
        n = self._nidaqmx
        c = n.constants
        try:
            self._ai = n.Task()
            self._ai.ai_channels.add_ai_voltage_chan(
                self._read_channel, terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0,
            )
            self._ai.ai_channels.add_ai_voltage_chan(
                self._response_channel, terminal_config=c.TerminalConfiguration.DIFF,
                min_val=-10.0, max_val=10.0,
            )
            self._ai.timing.cfg_samp_clk_timing(
                rate=self.sample_rate, sample_mode=c.AcquisitionType.CONTINUOUS,
            )
            self._ai.in_stream.input_buf_size = max(
                self._ai.in_stream.input_buf_size, int(self.sample_rate * 2),
            )
            self._ai.in_stream.overwrite_mode = c.OverwriteMode.OVERWRITE_UNREAD_SAMPLES
            self._ai.in_stream.read_relative_to = c.ReadRelativeTo.MOST_RECENT_SAMPLE
            self._ai.in_stream.read_offset = -self._period
            self._ao = n.Task()
            self._ao.ao_channels.add_ao_voltage_chan(self._command_channel)
            self._ao.timing.cfg_samp_clk_timing(
                rate=self.sample_rate, sample_mode=c.AcquisitionType.CONTINUOUS,
                samps_per_chan=self._period,
            )
            self._ao.out_stream.regen_mode = c.RegenerationMode.ALLOW_REGENERATION
            self._ao.triggers.start_trigger.cfg_dig_edge_start_trig(
                self._trigger, trigger_edge=c.Edge.RISING,
            )
            self._ao.write(self._wave, auto_start=False)
            self._ao.start()  # arm the trigger before the AI emits its StartTrigger
            self._ao_started = True
            self._ai.start()
            self._ai_started = True
        except BaseException:
            self._close_tasks()
            raise

    def _close_tasks(self) -> None:
        ai, ao = self._ai, self._ao
        started = ((ao, self._ao_started), (ai, self._ai_started))
        self._ai = self._ao = None
        self._ai_started = self._ao_started = False
        first_error = None
        for task, running in started:
            if task is None:
                continue
            if running:
                try:
                    task.stop()
                except Exception as exc:
                    first_error = first_error or exc
            try:
                task.close()
            except Exception as exc:
                first_error = first_error or exc
        if ao is not None:
            # A stopped hardware-timed AO can retain its last (high) sample.
            # Drive zero with a fresh on-demand task before releasing the
            # device reservation, even after a read or setup error.
            zero = None
            try:
                zero = self._nidaqmx.Task()
                zero.ao_channels.add_ao_voltage_chan(self._command_channel)
                zero.write(0.0, auto_start=True)
            except Exception as exc:
                first_error = first_error or exc
            finally:
                if zero is not None:
                    try:
                        zero.close()
                    except Exception as exc:
                        first_error = first_error or exc
        if first_error is not None:
            raise RuntimeError('Could not stop, zero and close NI membrane-test tasks') from first_error

    def _process(self, raw: np.ndarray, *, cell_mode: bool | None = None) -> tuple:
        """Analyze one period with PatcherBot's R, Ra/Rm and decay-fit equations."""
        read = raw[0] * self.read_gain
        response = raw[1] * self.response_gain
        for arr in (read, response):
            arr.setflags(write=False)
        # Include the wraparound edge; a MOST_RECENT_SAMPLE window may start
        # anywhere within the pulse even though it contains one full period.
        changes = np.diff(read, append=read[0])
        rise = int(np.argmax(changes)) + 1
        fall = int(np.argmin(changes)) + 1
        width = (fall - rise) % self._period
        if width < 8 or width >= self._period - 8:
            raise ValueError('NI command readback has no complete pulse')
        baseline_width = min(100, self._period - width)
        high_idx = (np.arange(width) + rise) % self._period
        low_idx = (np.arange(baseline_width) + rise - baseline_width) % self._period
        # Advanced indexing also aligns a pulse crossing the circular buffer
        # boundary, with no additional copy of the full 2-channel snapshot.
        high_read = read[high_idx]
        high_response = response[high_idx]
        low_read = read[low_idx]
        low_response = response[low_idx]
        delta_v = float(np.mean(high_read) - np.mean(low_read))
        if delta_v <= self.command_amplitude * 0.1:
            raise ValueError('NI command readback does not match a positive test pulse')
        # PatcherBot's total resistance uses the two means on either side of
        # the response mean over the entire read window.
        response_mean = float(np.mean(response))
        below = response[response < response_mean]
        above = response[response > response_mean]
        if not below.size or not above.size:
            raise ValueError('NI response has no resolvable membrane-test current')
        delta_i = float(np.mean(above) - np.mean(below))
        resistance = delta_v / delta_i
        if not (self._cell_mode.is_set() if cell_mode is None else cell_mode):
            return self._time_axis, read, response, raw, resistance, None, None, None
        # Preserve PatcherBot's mono-exponential model I = m exp(-t ms) + b,
        # robust soft_l1 least-squares bounds (tau in [0.05, 100] ms), and
        # circuit formulas, but reject nonphysical / underdetermined fits.
        peak = int(np.argmax(high_response))
        if peak >= width - 8:
            raise ValueError('No decay interval after membrane current peak')
        i_prev = float(np.mean(low_response))
        i_ss = float(np.mean(high_response[-max(8, width // 4):]))
        i_peak = float(high_response[min(peak + 1, width - 1)])
        fit_y = high_response[peak:] * 1e9  # amperes -> nA for conditioning
        fit_x = self._time_axis[:fit_y.size] * 1000
        guess = np.array([
            (float(high_response[peak]) - i_ss) * 1e9,
            max(1 / max(peak * (1000 / self.sample_rate), 0.1), 0.01),
            i_ss * 1e9,
        ])
        bounds = ([-10.0, 0.01, -10.0], [10.0, 20.0, 10.0])
        guess = np.clip(guess, np.nextafter(bounds[0], math.inf), np.nextafter(bounds[1], -math.inf))

        def residual(params):
            m, rate, b = params
            return m * np.exp(-rate * fit_x) + b - fit_y

        def jacobian(params):
            m, rate, _ = params
            exp = np.exp(-rate * fit_x)
            return np.column_stack((exp, -m * fit_x * exp, np.ones_like(fit_x)))

        result = least_squares(
            residual, guess, jac=jacobian, bounds=bounds,
            loss='soft_l1', f_scale=1.0, max_nfev=400,
        )
        if not result.success or not np.isfinite(result.x).all() or result.x[0] <= 0:
            raise ValueError('Membrane mono-exponential fit failed')
        tau_s = 1e-3 / result.x[1]
        di_peak, di_ss = i_peak - i_prev, i_ss - i_prev
        if di_peak <= 0 or di_ss <= 0:
            raise ValueError('Membrane fit yields nonpositive test-pulse currents')
        ra = delta_v / di_peak
        rm = delta_v / di_ss - ra
        if ra <= 0 or rm <= 0:
            raise ValueError('Membrane fit yields nonpositive access or membrane resistance')
        cm = tau_s * (1 / ra + 1 / rm)
        if not all(math.isfinite(v) and v > 0 for v in (resistance, ra, rm, cm)):
            raise ValueError('Membrane fit yields invalid electrical quantities')
        return self._time_axis, read, response, raw, resistance, ra, rm, cm
