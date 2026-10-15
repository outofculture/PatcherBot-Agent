"""Scientifica XYZ stage with independent streamed Arduino Z quadrature feedback."""

import threading
import time

import numpy as np

from acq4.devices.Stage import MoveFuture, Stage


class PatcherBotEncoderStage(Stage):
    """Correct Z using measured quadrature counts after each Scientifica move.

    ACQ4 positions and speeds are SI metres and metres/second. ``stageUnitsPerEncoderPulse``
    is required in micrometres/count because the two supplied Arduino firmware
    variants use different quadrature decoding and have conflicting calibrations.
    Arduino accepts no commands: it streams ``ENC\\n<count>\\n`` or ``<count>\\n``.
    An injected ``stageTransport`` must implement getPos/moveTo/stop/close as
    ACQ4's Scientifica driver; ``encoderTransport`` has pyserial's read/in_waiting/close.
    """

    def __init__(self, manager, config, name):
        if 'stageUnitsPerEncoderPulse' not in config:
            raise ValueError('Encoder stage requires calibrated stageUnitsPerEncoderPulse (µm/count)')
        self._metres_per_count = float(config['stageUnitsPerEncoderPulse']) * 1e-6
        if not np.isfinite(self._metres_per_count) or self._metres_per_count == 0:
            raise ValueError('stageUnitsPerEncoderPulse must be finite and nonzero')
        self._tolerance = float(config.get('correctionTolerance', 2e-6))
        self._max_retries = int(config.get('correctionMaxRetries', 5))
        self._encoder_timeout = float(config.get('encoderTimeout', 2.0))
        if (not np.isfinite(self._tolerance) or self._tolerance <= 0
                or self._max_retries < 1 or not np.isfinite(self._encoder_timeout)
                or self._encoder_timeout <= 0):
            raise ValueError('Encoder tolerance, retries, and timeout must be positive')
        self._samples = threading.Condition()
        self._encoder_count = None
        self._last_sample_time = None
        self._encoder_seq = 0
        self._encoder_error = None
        self._buffer = bytearray()
        self._expect_count = False
        self._closing = threading.Event()
        self._motion_lock = threading.Lock()
        self._driver = config.get('stageTransport')
        if self._driver is None:
            from acq4.drivers.Scientifica import Scientifica

            self._driver = Scientifica(
                port=config['port'], name=config.get('controllerName', name),
                baudrate=config.get('baudrate'), ctrl_version=config.get('version', 2),
            )
        self._encoder = config.get('encoderTransport')
        stage_initialized = False
        try:
            if self._encoder is None:
                import serial

                self._encoder = serial.Serial(config['encoderPort'], baudrate=115200, timeout=0.1)
            super().__init__(manager, config, name)
            stage_initialized = True
            self._reader = threading.Thread(target=self._read_encoder, daemon=True, name=f'{name}-encoder')
            self._reader.start()
            self._encoder_value(timeout=self._encoder_timeout)
            self._getPosition()
        except BaseException:
            self._closing.set()
            if hasattr(self, '_reader') and self._reader.is_alive():
                self._reader.join(timeout=self._encoder_timeout)
            if stage_initialized:
                manager.globalHalt.remove_abort_callback(self.abortForHalt)
            if self._encoder is not None:
                self._encoder.close()
            self._driver.close()
            raise

    def axes(self):
        return 'x', 'y', 'z'

    def capabilities(self):
        return {'getPos': (True, True, True), 'setPos': (True, True, True),
                'limits': (False, False, False)}

    @property
    def positionUpdatesPerSecond(self):
        return self.config.get('encoderRateHz', 10.0)

    def _consume_encoder_bytes(self, data):
        self._buffer.extend(data)
        while b'\n' in self._buffer:
            line, _, remainder = self._buffer.partition(b'\n')
            self._buffer = bytearray(remainder)
            line = line.strip()
            if line == b'ENC':
                self._expect_count = True
                continue
            if not line:
                continue
            try:
                count = int(line)
            except ValueError:
                if self._expect_count:
                    raise ValueError(f'Invalid Arduino encoder sample: {line!r}')
                continue
            self._expect_count = False
            with self._samples:
                self._encoder_count = count
                self._last_sample_time = time.monotonic()
                self._encoder_seq += 1
                self._samples.notify_all()

    def _read_encoder(self):
        try:
            while not self._closing.is_set():
                count = self._encoder.in_waiting
                if count:
                    data = self._encoder.read(count)
                    self._consume_encoder_bytes(data)
                    if self._encoder_count is not None:
                        self._getPosition()
                else:
                    self._closing.wait(0.01)
        except BaseException as exc:
            with self._samples:
                self._encoder_error = exc
                self._samples.notify_all()
            self.stop(reason=f'Encoder stream failed: {exc}')

    def _encoder_value(self, after=None, timeout=None):
        if timeout is None:
            timeout = self._encoder_timeout
        with self._samples:
            available = self._samples.wait_for(
                lambda: self._encoder_error is not None or self._closing.is_set()
                or (self._encoder_count is not None and (after is None or self._encoder_seq > after)),
                timeout=timeout,
            )
            if self._encoder_error is not None:
                raise RuntimeError('Arduino encoder stream failed') from self._encoder_error
            if self._closing.is_set():
                raise RuntimeError('Encoder stage is closed')
            if not available:
                raise TimeoutError('No fresh Arduino encoder sample; stage motion is unsafe')
            if time.monotonic() - self._last_sample_time > self._encoder_timeout:
                raise TimeoutError('Arduino encoder sample is stale; stage motion is unsafe')
            return self._encoder_count * self._metres_per_count, self._encoder_seq

    def _getPosition(self):
        measured_z, _ = self._encoder_value()
        raw = self._driver.getPos()  # Scientifica reports micrometres.
        if len(raw) != 3:
            raise ValueError(f'Scientifica returned {len(raw)} axes, expected three')
        pos = [float(raw[0]) * 1e-6, float(raw[1]) * 1e-6, measured_z]
        if pos != self._lastPos:
            self.posChanged(pos)
        return pos

    def _move(self, pos, speed, linear, name=None, **kwds):
        if linear or kwds:
            raise ValueError('Encoder stage supports only independent XYZ position moves')
        with self._motion_lock:
            if self._lastMove is not None and not self._lastMove.is_done:
                self.stop(reason='Superseded by new move')
            current = self.getPosition(refresh=True)
            target = [current[i] if value is None else float(value) for i, value in enumerate(pos)]
            future = MoveFuture(self, target, self._interpretSpeed(speed), name=name)
            self._lastMove = future
            future.producerThread(lambda: self._perform_move(future), name=f'{self.name()}-correction').start()
            return future

    def _perform_move(self, future):
        try:
            target = future.targetPos
            z_move = abs(target[2] - future.startPos[2]) > self._tolerance
            for attempt in range(self._max_retries if z_move else 1):
                if future.is_done or self._closing.is_set():
                    return
                current_z, _ = self._encoder_value()
                raw_pos_um = self._driver.getPos()
                raw_target_um = [float(target[0]) * 1e6, float(target[1]) * 1e6,
                                 float(raw_pos_um[2]) + (float(target[2]) - current_z) * 1e6]
                if not z_move:
                    raw_target_um[2] = float(raw_pos_um[2])
                request = self._driver.moveTo(raw_target_um, speed=future.speed * 1e6,
                                              name=f'{self.name()} corrected move')
                request.wait(timeout=None)
                if future.is_done:
                    return
                if z_move:
                    with self._samples:
                        completed_seq = self._encoder_seq
                    self._encoder_value(after=completed_seq)
                actual = self._getPosition()
                if all(abs(actual[i] - target[i]) <= self._tolerance for i in range(3)):
                    future.resolve()
                    return
            raise RuntimeError(f'Encoder-corrected stage did not reach {list(target)} m '
                               f'after {self._max_retries if z_move else 1} attempt(s)')
        except BaseException as exc:
            future.fail(exc)
            self._driver.stop(reason=f'Encoder stage move failed: {exc}')

    def stop(self, reason=None):
        move = self._lastMove
        if move is not None and not move.is_done:
            move.fail(RuntimeError(reason or 'Encoder stage stopped'))
        self._driver.stop(reason=reason or 'Encoder stage stopped')

    def abort(self):
        self.stop(reason='Encoder stage aborted')

    def close(self):
        self.quit()

    def quit(self):
        if self._closing.is_set():
            return
        try:
            super().quit()
        finally:
            self._closing.set()
            with self._samples:
                self._samples.notify_all()
            self._reader.join(timeout=self._encoder_timeout)
            self._encoder.close()
            self._driver.close()
