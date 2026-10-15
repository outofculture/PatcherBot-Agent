"""Hardware-free coverage of NI task configuration, safety, and fit semantics."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from patcherbot_acq4.acquisition import ContinuousMembraneTest


CONFIG = dict(
    readChannel='/Dev1/ai0', responseChannel='/Dev1/ai1', commandChannel='/Dev1/ao0',
)


def pulse_trace():
    """A 10 mV step, 25 MΩ access, 75 MΩ membrane, ~50 pF equivalent."""
    sample_rate = 50_000
    t = np.arange(1250) / sample_rate
    volts = np.zeros(1250)
    amps = np.zeros(1250)
    start, end = 25, 650
    volts[start:end] = .01
    amps[start:end] = 100e-12 + 300e-12 * np.exp(-(t[start:end] - t[start]) / .001)
    return np.vstack((volts / .02, amps / 2e-9))


class Halt:
    def __init__(self):
        self.callbacks = []
        self.halted = False

    def check(self):
        if self.halted:
            raise RuntimeError('Rig halted')

    def add_abort_callback(self, cb, name):
        self.callbacks.append(cb)

    def remove_abort_callback(self, cb):
        self.callbacks.remove(cb)

    def halt(self):
        self.halted = True
        for callback in tuple(self.callbacks):
            callback()


class Manager:
    def __init__(self):
        self.globalHalt = Halt()
        self.owner = None
        self.reserve_count = 0
        self.release_count = 0

    def reserveDevices(self, devices, reserver):
        assert len(devices) == 1
        manager = self

        class Reservation:
            def __enter__(self):
                assert manager.owner is None
                manager.owner = threading.get_ident()
                manager.reserve_count += 1

            def __exit__(self, *_):
                assert manager.owner == threading.get_ident()
                manager.owner = None
                manager.release_count += 1

        return Reservation()


class FakeNidaqmx:
    def __init__(self, trace):
        self.trace = trace
        self.tasks = []
        self.fail_read = False
        self.fail_start = False
        self.starts = []
        self.read_count = 0
        self.constants = SimpleNamespace(
            TerminalConfiguration=SimpleNamespace(DIFF='diff'),
            AcquisitionType=SimpleNamespace(CONTINUOUS='continuous'),
            OverwriteMode=SimpleNamespace(OVERWRITE_UNREAD_SAMPLES='overwrite'),
            ReadRelativeTo=SimpleNamespace(MOST_RECENT_SAMPLE='recent'),
            RegenerationMode=SimpleNamespace(ALLOW_REGENERATION='regen'),
            Edge=SimpleNamespace(RISING='rising'),
        )

    def Task(self):
        task = FakeTask(self)
        self.tasks.append(task)
        return task


class FakeTask:
    def __init__(self, driver):
        self.driver = driver
        self.ai_channels = SimpleNamespace(add_ai_voltage_chan=self.add_ai)
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=self.add_ao)
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self.set_clock)
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(cfg_dig_edge_start_trig=self.trigger))
        self.in_stream = SimpleNamespace(input_buf_size=1000, overwrite_mode=None, read_relative_to=None, read_offset=None)
        self.out_stream = SimpleNamespace(regen_mode=None)
        self.channels = []
        self.wave = None
        self.clock = None
        self.start_trigger = None
        self.started = False
        self.closed = False

    def add_ai(self, channel, **kwargs):
        self.channels.append((channel, kwargs))

    def add_ao(self, channel):
        self.channels.append(channel)

    def set_clock(self, **kwargs):
        self.clock = kwargs

    def trigger(self, terminal, **kwargs):
        self.start_trigger = (terminal, kwargs)

    def write(self, wave, auto_start):
        self.wave = np.asarray(wave).copy()
        self.auto_start = auto_start

    def start(self):
        if self.driver.fail_start and self.wave is not None and self.wave.ndim == 1:
            raise RuntimeError('NI AO unavailable')
        self.driver.starts.append('AO' if self.wave is not None else 'AI')
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def read(self, number_of_samples_per_channel, timeout):
        assert number_of_samples_per_channel == 1250
        assert timeout <= 1.0
        self.driver.read_count += 1
        if self.driver.fail_read:
            raise RuntimeError('NI stream disconnected')
        # A full period can start at any AO phase. Exercise wrapping the
        # analysis window without assuming read() is locked to the pulse edge.
        return np.roll(self.driver.trace, 850, axis=1).tolist()


def instrument(trace=None, **settings):
    manager = Manager()
    driver = FakeNidaqmx(pulse_trace() if trace is None else trace)
    daq = SimpleNamespace(exclusiveDevice='Dev1')
    stream = ContinuousMembraneTest(manager, daq, {**CONFIG, **settings}, nidaqmx_factory=lambda: driver)
    return stream, manager, driver


def wait_sample(stream):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            return stream.latest()
        except RuntimeError:
            time.sleep(.005)
    pytest.fail('no valid measurement arrived')


def test_synchronized_stream_si_fit_and_pause_resume():
    stream, manager, driver = instrument()
    stream.start()
    try:
        sample = wait_sample(stream)
        assert sample.acquisition_id >= 1
        assert sample.timestamp_monotonic_s <= time.monotonic()
        assert sample.raw.shape == (2, 1250)
        assert not sample.raw.flags.writeable
        assert not sample.read_v.flags.writeable
        assert sample.time_s[-1] == pytest.approx(1249 / 50000)
        assert sample.access_resistance_ohm is None  # bath: no circuit fit
        bath_id = sample.acquisition_id
        stream.set_cell_mode(True)
        with pytest.raises(RuntimeError, match='No valid'):
            stream.latest()
        sample = wait_sample(stream)
        assert sample.acquisition_id > bath_id
        assert sample.access_resistance_ohm == pytest.approx(25e6, rel=.15)
        assert sample.membrane_resistance_ohm == pytest.approx(75e6, rel=.15)
        assert sample.membrane_capacitance_farad == pytest.approx(50e-12, rel=.25)
        assert sample.resistance_ohm > 0
        ai, ao = driver.tasks[:2]
        assert ai.channels[0][0] == 'Dev1/ai0'
        assert ai.channels[1][0] == 'Dev1/ai1'
        assert ai.clock['rate'] == ao.clock['rate'] == 50_000
        assert ai.in_stream.read_relative_to == 'recent'
        assert ai.in_stream.read_offset == -1250
        assert ai.in_stream.overwrite_mode == 'overwrite'
        assert ao.out_stream.regen_mode == 'regen'
        assert driver.starts[:2] == ['AO', 'AI']
        assert ao.start_trigger == ('/Dev1/ai/StartTrigger', {'trigger_edge': 'rising'})
        assert ao.wave[0] == 0
        assert np.all(ao.wave[25:650] == .5)
        assert ao.wave[650:].max() == 0
        stream.pause()
        assert manager.owner is None
        assert manager.release_count == 1
        assert all(task.closed for task in driver.tasks)
        assert float(driver.tasks[2].wave) == 0
        with pytest.raises(RuntimeError, match='No valid'):
            stream.latest()
        stream.resume()
        resumed = wait_sample(stream)
        assert resumed.acquisition_id > sample.acquisition_id
        assert manager.reserve_count == 2
    finally:
        stream.stop()
    assert manager.owner is None
    assert manager.release_count == 2
    assert float(driver.tasks[-1].wave) == 0
    assert not manager.globalHalt.callbacks
    with pytest.raises(RuntimeError, match='No valid'):
        stream.latest()


def test_mode_switch_drops_in_flight_bath_measurement():
    stream, _, _ = instrument()
    first_entered, release_first = threading.Event(), threading.Event()
    second_entered, release_second = threading.Event(), threading.Event()
    process = stream._process
    calls = 0

    def controlled_process(raw, *, cell_mode=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert release_first.wait(2)
        elif calls == 2:
            second_entered.set()
            assert release_second.wait(2)
        return process(raw, cell_mode=cell_mode)

    stream._process = controlled_process
    stream.start()
    try:
        assert first_entered.wait(2)
        stream.set_cell_mode(True)
        with pytest.raises(RuntimeError, match='No valid'):
            stream.latest()
        release_first.set()
        assert second_entered.wait(2)
        with pytest.raises(RuntimeError, match='No valid'):
            stream.latest()
        release_second.set()
        assert wait_sample(stream).access_resistance_ohm == pytest.approx(25e6, rel=.15)
    finally:
        release_first.set()
        release_second.set()
        stream.stop()


def test_invalid_fit_does_not_publish_plausible_zero():
    stream, manager, driver = instrument(trace=np.zeros((2, 1250)))
    stream.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with stream._sample_lock:
                if stream._error is not None:
                    break
            time.sleep(.005)
        with pytest.raises(RuntimeError, match='failed'):
            stream.latest()
    finally:
        stream.stop()
    assert manager.owner is None


def test_stream_error_releases_device_and_zeros_ao():
    stream, manager, driver = instrument()
    driver.fail_read = True
    stream.start()
    try:
        deadline = time.monotonic() + 2
        while manager.owner is not None and time.monotonic() < deadline:
            time.sleep(.005)
        with pytest.raises(RuntimeError, match='failed'):
            stream.latest()
    finally:
        stream.stop()
    assert manager.owner is None
    assert manager.reserve_count == manager.release_count == 1
    assert float(driver.tasks[-1].wave) == 0


def test_startup_error_closes_tasks_and_releases_reservation():
    stream, manager, driver = instrument()
    driver.fail_start = True
    with pytest.raises(RuntimeError, match='Cannot start') as exc:
        stream.start()
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert 'NI AO unavailable' in str(exc.value.__cause__)
    assert manager.owner is None
    assert manager.reserve_count == manager.release_count == 1
    assert all(task.closed for task in driver.tasks)
    assert float(driver.tasks[-1].wave) == 0
    assert not manager.globalHalt.callbacks


def test_explicit_trigger_terminal_overrides_auto():
    stream, manager, driver = instrument(triggerTerminal='/cDAQ1/ai/StartTrigger')
    stream.start()
    try:
        assert driver.tasks[1].start_trigger[0] == '/cDAQ1/ai/StartTrigger'
    finally:
        stream.stop()


def test_global_halt_aborts_stream_before_unlocking():
    stream, manager, driver = instrument()
    stream.start()
    wait_sample(stream)
    manager.globalHalt.halt()
    assert manager.owner is None
    assert driver.tasks[-1].closed
    assert float(driver.tasks[-1].wave) == 0
    assert not manager.globalHalt.callbacks
    with pytest.raises(RuntimeError, match='No valid'):
        stream.latest()
    with pytest.raises(RuntimeError, match='halted'):
        stream.start()


def test_import_remains_lazy_and_missing_nidaqmx_fails_clearly():
    manager = Manager()

    def missing():
        raise ImportError('nidaqmx not installed')

    stream = ContinuousMembraneTest(
        manager, SimpleNamespace(exclusiveDevice='Dev1'), CONFIG, nidaqmx_factory=missing,
    )
    with pytest.raises(RuntimeError, match='requires the nidaqmx'):
        stream.start()
    assert manager.owner is None
    assert not manager.globalHalt.callbacks


def test_config_rejects_invalid_wave_and_wrong_physical_device():
    with pytest.raises(ValueError, match='integer'):
        instrument(sampleRate=50_001)
    with pytest.raises(ValueError, match='does not belong'):
        instrument(readChannel='/Dev2/ai0')
