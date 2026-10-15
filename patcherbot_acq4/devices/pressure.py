"""PatcherBot serial pressure boxes presented through ACQ4's pressure API.

All ACQ4 pressure values are Pascals; firmware calibration uses millibar.
``getPressure`` is the commanded setpoint, not a sensor measurement.
"""

import math
import threading

from acq4.devices.PressureControl.device import PressureControl


_CALIBRATIONS = {
    "Moscow": (1962, 3062 / 1100, 516.72, 0.3923),
    "IBB": (2048, 0.75, None, None),
    "Waynesboro": (1997, 49208 / 10000, 512.64, 0.3925),
}


class PatcherBotSerialPressure(PressureControl):
    """Serial pressure box (Moscow, IBB, or Waynesboro firmware).

    Config: ``variant``, ``channel``, ``port``, optional ``baudrate`` (default
    9600), ``readerPort``, ``readerBaudrate``, ``nativeZero``, ``nativePerMbar``,
    ``readerOffset``, ``readerScale``, ``timeout``. For injected transports use
    ``controllerSerial`` / ``readerSerial``; ownership is retained by the caller.
    No sensor reading is inferred when no reader is installed.
    """

    def __init__(self, manager, config, name):
        variant = config['variant']
        if variant not in _CALIBRATIONS:
            raise ValueError(f"Unsupported pressure variant {variant!r}; expected one of {tuple(_CALIBRATIONS)}")
        zero, scale, offset, reader_scale = _CALIBRATIONS[variant]
        self.channel = int(config['channel'])
        self.nativeZero = float(config.get('nativeZero', zero))
        self.nativePerMbar = float(config.get('nativePerMbar', scale))
        self.readerOffset = config.get('readerOffset', offset)
        self.readerScale = config.get('readerScale', reader_scale)
        if self.nativeZero <= 0 or self.nativePerMbar <= 0:
            raise ValueError('Pressure DAC calibration must be positive')
        if self.readerScale is not None and float(self.readerScale) == 0:
            raise ValueError('readerScale must be nonzero')
        self._io_lock = threading.RLock()
        self._reader_lock = threading.Lock()
        self._controller = None
        self._reader = None
        self._owns_controller = self._owns_reader = False
        self._raw_setpoint = None
        self._measured_pressure = None
        self._closed = False
        super().__init__(manager, config, name)
        self.sources = ('regulator', 'atmosphere')
        try:
            self._controller, self._owns_controller = self._open(config, 'controllerSerial', 'port', 'baudrate')
            if config.get('readerSerial') is not None or config.get('readerPort') is not None:
                if self.readerOffset is None or self.readerScale is None:
                    raise ValueError('Reader calibration readerOffset and readerScale is required')
                self._reader, self._owns_reader = self._open(config, 'readerSerial', 'readerPort', 'readerBaudrate')
            # Startup is vented, including after a previous process left a regulator active.
            self._setSource('atmosphere')
            self.source = 'atmosphere'
            self._setPressure(0)
            self.pressure = 0
        except Exception:
            self._close_ports()
            self.dm.globalHalt.remove_abort_callback(self.abort)
            raise

    @staticmethod
    def _open(config, injected_key, port_key, baud_key):
        supplied = config.get(injected_key)
        if supplied is not None:
            return supplied, False
        port = config.get(port_key)
        if not port:
            raise ValueError(f'{port_key} or {injected_key} is required for serial pressure control')
        import serial
        return serial.Serial(port=port, baudrate=int(config.get(baud_key, config.get('baudrate', 9600))),
                             timeout=float(config.get('timeout', 1)),
                             write_timeout=float(config.get('timeout', 1))), True

    def _command(self, command):
        with self._io_lock:
            if self._closed:
                raise RuntimeError('Pressure controller is closed')
            self._controller.write(command.encode('ascii'))
            self._controller.flush()

    def _setPressure(self, pressure):
        pressure = float(pressure)
        if not math.isfinite(pressure) or not self.minimum <= pressure <= self.maximum:
            raise ValueError(f'Pressure {pressure} Pa outside configured [{self.minimum}, {self.maximum}] Pa')
        raw = int(pressure / 100 * self.nativePerMbar + self.nativeZero)
        maximum_raw = 4095 if self.config['variant'] == 'IBB' else int(self.nativeZero * 2)
        if not 0 <= raw <= maximum_raw:
            raise ValueError(f'Pressure {pressure} Pa is outside calibrated DAC range 0..{maximum_raw}')
        self._command(f'set {self.channel} {raw}\n')
        self._raw_setpoint = raw

    def _setSource(self, source):
        if source not in self.sources:
            raise ValueError(f'Unsupported pressure source {source!r}')
        self._command(f'{"switchAtm" if source == "atmosphere" else "switchP"} {self.channel}\n')

    def getPressure(self):
        """Return the last requested regulator setpoint in Pa (not measured pressure)."""
        return self.pressure

    def getSource(self):
        return self.source

    def measurePressure(self):
        """Request the independent reader's actual sensor value, in Pa.

        The reader speaks ``R`` / ``S<number>E``. The legacy reader calibration
        maps this numeric value to millibar; it is not the DAC setpoint.
        """
        if self._reader is None:
            raise RuntimeError('No pressure sensor reader configured; getPressure is setpoint only')
        with self._reader_lock:
            self._reader.write(b'R')
            reply = self._reader.readline().decode('ascii').strip()
            if not reply.startswith('S') or not reply.endswith('E'):
                raise ValueError(f'Invalid pressure sensor response {reply!r}')
            measurement = (float(reply[1:-1]) - float(self.readerOffset)) / float(self.readerScale) * 100
            if not math.isfinite(measurement):
                raise ValueError(f'Non-finite pressure sensor response {reply!r}')
            self._measured_pressure = measurement
            return measurement

    def pulse(self, duration_ms):
        """Execute the firmware pulse within the configured duration limit."""
        if isinstance(duration_ms, bool) or int(duration_ms) != duration_ms:
            raise ValueError('Pulse duration must be an integral number of milliseconds')
        duration_ms = int(duration_ms)
        maximum = int(self.config.get('maximumPulseMs', 1000))
        if not 0 < duration_ms <= maximum:
            raise ValueError(f'Pulse duration must be between 1 and {maximum} ms')
        self.dm.globalHalt.check()
        self._command(f'pulse {self.channel} {duration_ms}\n')

    def abort(self):
        self.setPressure(source='atmosphere', pressure=0)

    def _close_ports(self):
        with self._io_lock:
            for port, owned in ((self._reader, self._owns_reader), (self._controller, self._owns_controller)):
                if port is not None and owned:
                    port.close()
            self._reader = self._controller = None
            self._closed = True

    def quit(self):
        try:
            if not self._closed:
                self.abort()
        finally:
            self._close_ports()
            super().quit()
