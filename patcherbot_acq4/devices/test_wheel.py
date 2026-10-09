"""Excelitas wheel digital step contract with injected ACQ4-style DAQ."""

from .optics import PatcherBotExcelitasWheel
from .test_hardware import Manager


class Daq:
    def __init__(self):
        self.values = []
        self.reservations = 0

    def reserve(self, *, block, reserver):
        self.reservations += 1
        return True

    def setChannelValue(self, channel, value, *, ignoreLock):
        assert ignoreLock is True
        self.values.append((channel, value))

    def release(self):
        self.reservations -= 1


def test_reserved_step_sequence(monkeypatch):
    monkeypatch.setattr('patcherbot_acq4.devices.optics.time.sleep', lambda interval: None)
    daq = Daq()
    wheel = PatcherBotExcelitasWheel(Manager(daq), {
        'daqDevice': 'daq', 'stepChannel': '/Dev1/port0/line0',
        'directionChannel': '/Dev1/port0/line1',
        'slots': {'0': 'first', '1': 'second', '2': 'third'},
        'initialSlot': 0, 'stepsPerPosition': 2, 'stepPeriodMs': 1,
        'pollInterval': None,
    }, 'wheel')
    wheel.setPosition(1)
    assert wheel.getPosition() == 1
    assert daq.reservations == 0
    assert daq.values == [('/Dev1/port0/line1', True),
                          ('/Dev1/port0/line0', True), ('/Dev1/port0/line0', False),
                          ('/Dev1/port0/line0', True), ('/Dev1/port0/line0', False),
                          ('/Dev1/port0/line0', False)]
    wheel.quit()
