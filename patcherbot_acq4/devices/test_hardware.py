"""Protocol checks with injected transports; no hardware is opened."""

import pytest

from .optics import PatcherBotLumencor, PatcherBotOlympusLamp
from .pressure import PatcherBotSerialPressure


class Halt:
    def __init__(self):
        self.callbacks = []

    def add_abort_callback(self, callback, name):
        self.callbacks.append(callback)

    def remove_abort_callback(self, callback):
        self.callbacks.remove(callback)

    def check(self):
        pass


class Manager:
    def __init__(self, daq=None):
        self.globalHalt = Halt()
        self.daq = daq

    def declareInterface(self, *args):
        pass

    def getDevice(self, name):
        assert name == 'daq'
        return self.daq


class Serial:
    def __init__(self, responses=()):
        self.writes = []
        self.responses = iter(responses)

    def write(self, message):
        self.writes.append(message)

    def flush(self):
        pass

    def readline(self):
        return next(self.responses)


class Spectra:
    def __init__(self):
        self.commands = []

    def setEnabled(self, channels):
        self.commands.append(('enabled', set(channels)))

    def setIntensity(self, channels, intensity):
        self.commands.append(('intensity', list(channels), intensity))


def test_pressure_commands_sensor_and_vent():
    serial = Serial()
    reader = Serial([b'S555.95E\n'])
    manager = Manager()
    dev = PatcherBotSerialPressure(manager, {'variant': 'Moscow', 'channel': 2,
                                              'controllerSerial': serial, 'readerSerial': reader,
                                              'regulatorSettlingTime': 0}, 'pressure')
    dev.setPressure(source='regulator', pressure=1000)
    assert dev.getPressure() == 1000
    assert dev.measurePressure() == pytest.approx(10000)
    assert reader.writes == [b'R']
    dev.pulse(30)
    manager.globalHalt.callbacks[0]()
    assert dev.getSource() == 'atmosphere'
    assert serial.writes == [b'switchAtm 2\n', b'set 2 1962\n', b'set 2 1989\n',
                             b'switchP 2\n', b'pulse 2 30\n', b'switchAtm 2\n', b'set 2 1962\n']
    dev.quit()

def test_olympus_shutter_cube_and_cleanup():
    transport = Serial([b'1LOG IN\n', b'1SHUTTER IN\n', b'1SHUTTER IN\n',
                        b'+\n', b'+\n', b'1CUBE 2\n', b'1SHUTTER OUT\n', b'+\n'])
    dev = PatcherBotOlympusLamp(Manager(), {'serial': transport}, 'lamp')
    dev.setSourceActive('shutter', True)
    dev.setCube(2)
    dev.quit()
    assert b'1SHUTTER OUT\r\n' in transport.writes
    assert b'1CUBE 2\r\n' in transport.writes
    assert transport.writes[-1] == b'1SHUTTER IN\r\n'


def test_lumencor_native_driver_and_abort():
    driver = Spectra()
    manager = Manager()
    dev = PatcherBotLumencor(manager, {'driver': driver}, 'spectra')
    dev.setSourceBrightness('red', 0.5)
    dev.setSourceActive('red', True)
    manager.globalHalt.callbacks[0]()
    assert driver.commands == [('enabled', set()), ('intensity', ['red'], 128),
                               ('intensity', ['red'], 128), ('enabled', {'red'}),
                               ('enabled', set())]
    dev.quit()
