import threading

import numpy as np
import pytest

from patcherbot_acq4.devices.pco import PatcherBotPcoCamera


class FakeRecorder:
    def __init__(self):
        self.count = 0

    def get_status(self):
        return {'dwProcImgCount': self.count}


class FakePco:
    def __init__(self):
        self.configuration = None
        self.rec = FakeRecorder()
        self.image_data = np.arange(16, dtype=np.uint16).reshape((4, 4))
        self.calls = []

    def get_exposure_time(self):
        return self.configuration['exposure time']

    def record(self, **kwargs):
        self.calls.append(('record', kwargs))

    def wait_for_first_image(self):
        self.calls.append(('wait',))

    def image(self, **kwargs):
        self.calls.append(('image', kwargs))
        return self.image_data, {}

    def stop(self):
        self.calls.append(('stop',))

    def close(self):
        self.calls.append(('close',))


def fake_camera():
    camera = PatcherBotPcoCamera.__new__(PatcherBotPcoCamera)
    camera.config = {'cameraFactory': FakePco}
    camera.camLock = threading.RLock()
    camera.setupCamera()
    return camera


def test_ring_delivers_permanent_latest_copy_and_host_arrival_time():
    camera = fake_camera()
    camera.startCamera()
    camera._camera.rec.count = 1
    frames = camera.newFrames()
    assert len(frames) == 1
    frame = frames[0]
    assert frame['id'] == 0
    assert frame['recorderFrameCount'] == 1
    assert frame['timeSource'] == 'hostRetrieval'
    assert 'exposeStartTime' not in frame
    assert camera._camera.calls[-1] == ('image', {'image_number': 0xFFFFFFFF})
    camera._camera.image_data[:] = 0
    assert frame['data'][0, 1] == 1
    assert camera.newFrames() == []
    camera._camera.rec.count = 4  # lost intermediate ring frames are not fabricated
    assert [frame['id'] for frame in camera.newFrames()] == [3]
    camera.stopCamera()
    assert camera.newFrames() == []
    assert camera._camera.calls[-1] == ('stop',)


def test_roi_exposure_are_converted_only_at_sdk_boundary():
    camera = fake_camera()
    camera.startCamera()
    applied, restarted = camera.setParams({'region': (10, 20, 64, 32), 'exposure': .035})
    assert restarted
    assert applied == {'region': (10, 20, 64, 32), 'exposure': .035}
    assert camera._camera.configuration['roi'] == (11, 21, 74, 52)
    assert camera._camera.configuration['exposure time'] == .035
    assert camera._camera.calls[-2][0] == 'record'
    with pytest.raises(ValueError, match='outside'):
        camera.setParams({'region': (2000, 0, 64, 32)})
    with pytest.raises(ValueError, match='triggerMode'):
        camera.setParams({'triggerMode': 'TriggerStart'})


def test_region_component_gui_params_and_safe_close():
    camera = fake_camera()
    camera.startCamera()
    params, restarted = camera.setParams({'regionX': 0, 'regionW': 64}, autoRestart=False)
    assert restarted
    assert params == {'regionX': 0, 'regionW': 64}
    assert camera._camera.configuration['roi'] == (1, 385, 64, 1664)
    assert camera.listParams(['regionW'])['regionW'][0] == (1, 2048)
    sdk = camera._camera
    camera.close()
    camera.close()
    assert sdk.calls[-1] == ('close',)
    assert sum(call[0] == 'close' for call in sdk.calls) == 1
