import threading
import time

import pytest

from patcherbot_acq4.devices.encoder import PatcherBotEncoderStage


class FakeDriver:
    def __init__(self, samples):
        self.xyz_um = [0, 0, 10]
        self.samples = iter(samples)
        self.commands = []
        self.stage = None

    def getPos(self):
        return self.xyz_um[:]

    def moveTo(self, pos, speed, name):
        self.commands.append((list(pos), speed))
        return FakeRequest(self, pos)

    def stop(self, reason):
        self.commands.append(('stop', reason))

    def close(self):
        self.commands.append(('close',))


class FakeRequest:
    def __init__(self, driver, position):
        self.driver = driver
        self.position = position

    def wait(self, timeout):
        self.driver.xyz_um = list(self.position)
        count = next(self.driver.samples)
        threading.Timer(
            .001, lambda: self.driver.stage._consume_encoder_bytes(f'ENC\n{count}\n'.encode())
        ).start()


class FakeFuture:
    speed = 2e-4
    targetPos = (0, 0, 12e-6)
    startPos = (0, 0, 10e-6)
    is_done = False

    def resolve(self):
        self.is_done = True

    def fail(self, exc):
        raise exc


def fake_stage(samples=(11, 12)):
    stage = PatcherBotEncoderStage.__new__(PatcherBotEncoderStage)
    stage._samples = threading.Condition()
    stage._buffer = bytearray()
    stage._expect_count = False
    stage._encoder_count = None
    stage._encoder_seq = 0
    stage._encoder_error = None
    stage._closing = threading.Event()
    stage._encoder_timeout = .1
    stage._metres_per_count = 1e-6
    stage._tolerance = .1e-6
    stage._max_retries = 2
    stage._lastPos = [0, 0, 10e-6]
    stage.posChanged = lambda pos: setattr(stage, '_lastPos', pos)
    stage.name = lambda: 'test stage'
    stage._driver = FakeDriver(samples)
    stage._driver.stage = stage
    stage._consume_encoder_bytes(b'ENC\n10\n')
    return stage


def test_encoder_protocol_partial_tagged_and_plain_lines():
    stage = fake_stage()
    stage._consume_encoder_bytes(b'ENC\n-')
    assert stage._encoder_value()[0] == pytest.approx(10e-6)
    stage._consume_encoder_bytes(b'2\n')
    assert stage._encoder_value()[0] == -2e-6
    stage._consume_encoder_bytes(b'4\n')
    assert stage._encoder_value()[0] == 4e-6
    with pytest.raises(ValueError, match='Invalid Arduino'):
        stage._consume_encoder_bytes(b'ENC\ninvalid\n')


def test_correction_uses_stage_delta_and_waits_for_fresh_encoder_count():
    stage = fake_stage()
    future = FakeFuture()
    stage._perform_move(future)
    assert future.is_done
    assert stage._driver.commands[0][0] == pytest.approx([0, 0, 12])
    assert stage._driver.commands[1][0] == pytest.approx([0, 0, 13])
    assert [cmd[1] for cmd in stage._driver.commands] == [200, 200]
    assert stage._getPosition()[2] == pytest.approx(12e-6)


def test_missing_fresh_count_aborts_instead_of_assuming_motion_succeeded():
    stage = fake_stage()
    with pytest.raises(TimeoutError, match='fresh Arduino'):
        stage._encoder_value(after=stage._encoder_seq)
    stage._last_sample_time = time.monotonic() - 1
    with pytest.raises(TimeoutError, match='stale'):
        stage._encoder_value()


def test_encoder_transport_failure_stops_stage_and_is_not_silenced():
    class BrokenEncoder:
        @property
        def in_waiting(self):
            raise OSError('encoder disconnected')

    stage = fake_stage()
    stage._encoder = BrokenEncoder()
    stage._lastMove = None
    stage._read_encoder()
    assert stage._driver.commands == [('stop', 'Encoder stream failed: encoder disconnected')]
    with pytest.raises(RuntimeError, match='Arduino encoder stream failed'):
        stage._encoder_value()
