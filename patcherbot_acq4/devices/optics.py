"""ACQ4 optics adapters for Olympus, Excelitas and Lumencor hardware."""

import threading
import time

from acq4.devices.LightSource import LightSource
from acq4.devices.FilterWheel.filterwheel import FilterWheel, FilterWheelFuture


class PatcherBotOlympusLamp(LightSource):
    """Olympus UCB shutter plus cube selector.

    Config: ``port`` or injected ``serial``, ``baudrate`` (default 19200),
    ``cubeSlots`` (default 6), ``cubeCommand`` (CUBE or MU), ``address`` (1).
    Port format is 8E2 RTS/CTS as in the PatcherBot driver.
    """

    def __init__(self, manager, config, name):
        self._io_lock = threading.RLock()
        self._serial = config.get('serial')
        self._owns_serial = self._serial is None
        self._closed = False
        self._slots = int(config.get('cubeSlots', 6))
        self._cube_command = config.get('cubeCommand', 'CUBE')
        self._address = str(config.get('address', '1'))
        if self._slots <= 0 or self._cube_command not in ('CUBE', 'MU'):
            raise ValueError('cubeSlots must be positive and cubeCommand must be CUBE or MU')
        super().__init__(manager, config, name)
        try:
            if self._serial is None:
                import serial
                self._serial = serial.Serial(port=config['port'], baudrate=int(config.get('baudrate', 19200)),
                                             bytesize=serial.EIGHTBITS, parity=serial.PARITY_EVEN,
                                             stopbits=serial.STOPBITS_TWO, rtscts=True,
                                             timeout=1, write_timeout=1)
            if not self._command('LOG?').endswith('IN') and not self._command('LOG IN').endswith('+'):
                raise RuntimeError('Olympus UCB login failed')
            self.addSource('shutter', {'active': self.shutterOpen(), 'adjustableBrightness': False})
            manager.globalHalt.add_abort_callback(self.abort, name=f'{name}.abort')
        except Exception:
            if self._owns_serial and self._serial is not None:
                self._serial.close()
            raise

    def _command(self, tail):
        with self._io_lock:
            if self._closed:
                raise RuntimeError('Olympus UCB is closed')
            self._serial.write(f'{self._address}{tail}\r\n'.encode('ascii'))
            reply = self._serial.readline()
            if not reply:
                raise TimeoutError(f'No Olympus UCB response to {tail!r}')
            result = reply.decode('ascii', errors='replace').strip()
            if result.startswith('!,E'):
                raise RuntimeError(f'Olympus UCB error: {result}')
            return result

    def shutterOpen(self):
        status = self._command('SHUTTER?').split()[-1]
        if status not in ('IN', 'OUT'):
            raise ValueError(f'Unknown Olympus shutter status {status!r}')
        return status == 'OUT'

    def abort(self):
        self.setSourceActive('shutter', False)

    def setSourceActive(self, name, active):
        if name != 'shutter':
            raise KeyError(name)
        if active:
            self.dm.globalHalt.check()
        with self._io_lock:
            if self.shutterOpen() != bool(active):
                self._command('SHUTTER OUT' if active else 'SHUTTER IN')
            self.sourceConfigs[name]['active'] = bool(active)
            self.sigLightChanged.emit(self, name)

    def getCube(self):
        with self._io_lock:
            response = self._command(f'{self._cube_command}?')
            # The controller may first echo an acknowledgement before returning position.
            while response.strip(' +') in ('', 'CUBE', 'MU'):
                response = self._serial.readline().decode('ascii', errors='replace').strip()
                if not response:
                    raise TimeoutError('No Olympus cube position response')
                if response.startswith('!,E'):
                    raise RuntimeError(f'Olympus cube error: {response}')
            for token in response.replace(',', ' ').split():
                if token.isdigit():
                    return int(token)
            raise ValueError(f'Unknown Olympus cube position response {response!r}')

    def setCube(self, slot):
        slot = int(slot)
        if not 1 <= slot <= self._slots:
            raise ValueError(f'Cube slot must be 1..{self._slots}')
        self.dm.globalHalt.check()
        with self._io_lock:
            self._command(f'{self._cube_command} {slot}')
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.getCube() == slot:
                    return
                time.sleep(0.3)
        raise TimeoutError(f'Olympus cube did not reach slot {slot}')

    def quit(self):
        try:
            if not self._closed:
                self.setSourceActive('shutter', False)
        finally:
            self.dm.globalHalt.remove_abort_callback(self.abort)
            with self._io_lock:
                self._closed = True
                if self._owns_serial and self._serial is not None:
                    self._serial.close()
            super().quit()


class PatcherBotLumencor(LightSource):
    """Spectra X source driven by ACQ4's native SpectraXDriver.

    Config: ``port`` or injected ``driver`` implementing setEnabled,
    setIntensity, close. Native driver supports red/green/cyan/uv/blue/teal.
    """

    CHANNELS = ('red', 'green', 'cyan', 'uv', 'blue', 'teal')

    def __init__(self, manager, config, name):
        self._io_lock = threading.RLock()
        self._driver = config.get('driver')
        self._owns_driver = self._driver is None
        self._active = set()
        self._brightness = {channel: 1.0 for channel in self.CHANNELS}
        super().__init__(manager, config, name)
        if self._driver is None:
            from acq4.drivers.Lumencor.spectrax import SpectraXDriver
            self._driver = SpectraXDriver(config['port'])
        try:
            self._driver.setEnabled([])
            for channel in self.CHANNELS:
                self.addSource(channel, {'active': False, 'adjustableBrightness': True,
                                         **{k: v for k, v in config.get('sources', {}).get(channel, {}).items()
                                            if k not in ('active', 'adjustableBrightness')}})
            manager.globalHalt.add_abort_callback(self.abort, name=f'{name}.abort')
        except Exception:
            if self._owns_driver:
                self._driver.close()
            raise

    def setSourceActive(self, name, active):
        if name not in self.CHANNELS:
            raise KeyError(name)
        if active:
            self.dm.globalHalt.check()
        with self._io_lock:
            if active:
                self._driver.setIntensity([name], round(self._brightness[name] * 255))
            selected = self._active | {name} if active else self._active - {name}
            self._driver.setEnabled(selected)
            self._active = selected
            self.sourceConfigs[name]['active'] = bool(active)
            self.sigLightChanged.emit(self, name)

    def getSourceBrightness(self, name):
        return self._brightness[name]

    def setSourceBrightness(self, name, value):
        if name not in self.CHANNELS:
            raise KeyError(name)
        if not 0 <= value <= 1:
            raise ValueError('Brightness must be between 0 and 1')
        self.dm.globalHalt.check()
        with self._io_lock:
            self._driver.setIntensity([name], round(value * 255))
            self._brightness[name] = value
            self.sigLightChanged.emit(self, name)

    def abort(self):
        with self._io_lock:
            self._driver.setEnabled([])
            self._active.clear()
            for name in self.CHANNELS:
                self.sourceConfigs[name]['active'] = False
                self.sigLightChanged.emit(self, name)

    def quit(self):
        try:
            self.abort()
        finally:
            self.dm.globalHalt.remove_abort_callback(self.abort)
            if self._owns_driver:
                self._driver.close()
            super().quit()


class PatcherBotExcelitasWheel(FilterWheel):
    """Open-loop Excelitas step/direction wheel, using reserved ACQ4 DAQ lines.

    Config: ``daqDevice``, ``stepChannel``, ``directionChannel``, ``slots``
    (zero-based ACQ4 slot labels), ``initialSlot`` (required physical reference),
    ``stepsPerPosition`` (800), ``stepPeriodMs`` (1 per HIGH and LOW),
    ``directionHighForIncrease`` (True). No position sensor is present; the
    position becomes unknown after a partial/failed move.
    """

    def __init__(self, manager, config, name):
        if 'initialSlot' not in config:
            raise ValueError('Excelitas wheel requires initialSlot referencing the physical starting slot')
        if config['stepChannel'] == config['directionChannel']:
            raise ValueError('Step and direction channels must differ')
        self._position = int(config['initialSlot'])
        self._moving = False
        self._cancel = threading.Event()
        self._daq = manager.getDevice(config['daqDevice'])
        self._step = config['stepChannel']
        self._direction = config['directionChannel']
        self._steps_per_position = int(config.get('stepsPerPosition', 800))
        self._step_period = float(config.get('stepPeriodMs', 1)) / 1000
        self._direction_high = bool(config.get('directionHighForIncrease', True))
        if self._steps_per_position <= 0 or self._step_period <= 0:
            raise ValueError('stepsPerPosition and stepPeriodMs must be positive')
        # FilterWheel forwards config['ports'] to OptomechDevice, which requires
        # a list even when every slot is only a label rather than a filter device.
        wheel_config = dict(config)
        wheel_config.setdefault('ports', ['default'])
        super().__init__(manager, wheel_config, name)

    def _getPosition(self):
        return self._position

    def isMoving(self):
        return self._moving

    def _setPosition(self, pos):
        pos = int(pos)
        if not 0 <= pos < self.getPositionCount():
            raise ValueError(f'Wheel position {pos} out of range')
        if self._position is None:
            raise RuntimeError('Wheel position unknown after interrupted move; physically re-reference before moving')
        future = FilterWheelFuture(self, pos)
        if pos == self._position:
            future.resolve(pos)
            return future
        count = self.getPositionCount()
        forward = (pos - self._position) % count
        backward = forward - count
        delta = backward if abs(backward) < forward else forward
        self._cancel.clear()
        self._moving = True
        try:
            if not self._daq.reserve(block=True, reserver=f'{self.name()}.wheelMove'):
                raise RuntimeError('Cannot reserve DAQ for Excelitas wheel')
            try:
                self._daq.setChannelValue(self._direction, self._direction_high if delta > 0 else not self._direction_high,
                                          ignoreLock=True)
                for _ in range(abs(delta) * self._steps_per_position):
                    if self._cancel.is_set():
                        raise RuntimeError('Excelitas wheel move cancelled; position unknown')
                    self._daq.setChannelValue(self._step, True, ignoreLock=True)
                    time.sleep(self._step_period)
                    self._daq.setChannelValue(self._step, False, ignoreLock=True)
                    time.sleep(self._step_period)
                self._position = pos
            finally:
                self._daq.setChannelValue(self._step, False, ignoreLock=True)
                self._daq.release()
        except Exception:
            self._position = None
            raise
        finally:
            self._moving = False
        future.resolve(pos)
        return future

    def _stop(self):
        self._cancel.set()

    def setSpeed(self, speed):
        raise NotImplementedError('Excelitas step timing is fixed by stepPeriodMs configuration')

    def getSpeed(self):
        return self._step_period

    def quit(self):
        self._cancel.set()
        super().quit()
