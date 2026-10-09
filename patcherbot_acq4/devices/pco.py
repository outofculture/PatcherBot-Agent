"""PCO Panda ring-buffer camera adapted to the ACQ4 Camera API."""

import numpy as np

from acq4.devices.Camera import Camera
from acq4.util import ptime


class PatcherBotPcoCamera(Camera):
    """PCO ring camera; ``time`` is host retrieval time, not exposure time.

    Configuration accepts ``cameraFactory`` for injection, ``sensorSize`` in pixels,
    ``region`` as ACQ4 (x, y, width, height) zero-based pixels, and ``exposure``
    in seconds. The PCO SDK uses inclusive, one-based ROI corner coordinates.
    """

    LATEST_IMAGE = 0xFFFFFFFF

    def setupCamera(self):
        self._sensor_size = tuple(self.config.get('sensorSize', (2048, 2048)))
        if (len(self._sensor_size) != 2
                or any(not isinstance(size, int) or size <= 0 for size in self._sensor_size)):
            raise ValueError('sensorSize must be (width, height) in pixels')
        default_roi = (384, 384, 1280, 1280)
        if any(default_roi[i] + default_roi[i + 2] > self._sensor_size[i] for i in (0, 1)):
            default_roi = (0, 0, *self._sensor_size)
        self._region = tuple(self.config.get('region', default_roi))
        if (len(self._region) != 4
                or any(not isinstance(v, int) for v in self._region)
                or self._region[0] < 0 or self._region[1] < 0
                or self._region[2] < 1 or self._region[3] < 1
                or self._region[0] + self._region[2] > self._sensor_size[0]
                or self._region[1] + self._region[3] > self._sensor_size[1]):
            raise ValueError('PCO region is outside the configured sensor bounds')
        self._exposure = float(self.config.get('exposure', 0.020))
        if not np.isfinite(self._exposure) or self._exposure <= 0:
            raise ValueError('PCO exposure must be positive and finite seconds')
        factory = self.config.get('cameraFactory')
        if factory is None:
            import pco  # SDK is optional until actual device construction

            factory = pco.Camera
        self._camera = factory()
        self._running = False
        self._last_recorder_count = None
        self._next_frame_id = 0
        try:
            self._apply_config()
        except BaseException:
            self._camera.close()
            raise

    def _apply_config(self):
        x, y, width, height = self._region
        self._camera.configuration = {
            'exposure time': self._exposure,
            'roi': (x + 1, y + 1, x + width, y + height),
            'timestamp': 'off',
            'trigger': 'auto sequence',
            'acquire': 'auto',
            'metadata': 'on',
            'binning': (1, 1),
        }

    def listParams(self, params=None):
        width, height = self._sensor_size
        x, y, roi_width, roi_height = self._region
        specs = {
            'sensorSize': (None, False, True, []),
            'region': (None, False, True, []),
            'regionX': ((0, width - roi_width), True, True, ['regionW']),
            'regionY': ((0, height - roi_height), True, True, ['regionH']),
            'regionW': ((1, width - x), True, True, ['regionX']),
            'regionH': ((1, height - y), True, True, ['regionY']),
            'exposure': ((1e-6, 10.0), True, True, []),
            'binning': (None, False, True, []),
            'triggerMode': (['Normal'], False, True, []),
        }
        return specs if params is None else {p: specs[p] for p in params}

    def getParams(self, params=None):
        with self.camLock:
            x, y, width, height = self._region
            values = {
                'sensorSize': self._sensor_size,
                'region': self._region,
                'regionX': x,
                'regionY': y,
                'regionW': width,
                'regionH': height,
                'exposure': self._camera.get_exposure_time(),
                'binning': (1, 1),
                'triggerMode': 'Normal',
            }
            return values if params is None else {p: values[p] for p in params}

    def setParams(self, params, autoRestart=True, autoCorrect=True):
        changes = dict(params)
        allowed = {'region', 'regionX', 'regionY', 'regionW', 'regionH',
                   'exposure', 'binning', 'triggerMode', 'sensorSize'}
        unknown = changes.keys() - allowed
        if unknown:
            raise KeyError(f'Unsupported PCO parameters: {sorted(unknown)}')
        for key, fixed in (('binning', (1, 1)), ('triggerMode', 'Normal'), ('sensorSize', self._sensor_size)):
            if key in changes and changes[key] != fixed:
                raise ValueError(f'PCO {key} must be {fixed!r}')
        if 'exposure' in changes:
            exposure = float(changes['exposure'])
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError('PCO exposure must be a positive finite number of seconds')
        region = list(changes.get('region', self._region))
        for index, key in enumerate(('regionX', 'regionY', 'regionW', 'regionH')):
            if key in changes:
                region[index] = changes[key]
        if len(region) != 4 or any(not isinstance(v, (int, np.integer)) for v in region):
            raise ValueError('PCO region must be four integer (x, y, width, height) values')
        x, y, width, height = region
        if x < 0 or y < 0 or width < 1 or height < 1 or x + width > self._sensor_size[0] or y + height > self._sensor_size[1]:
            raise ValueError('PCO region is outside the configured sensor bounds')
        region = tuple(region)
        changed = (region != self._region) or ('exposure' in changes and exposure != self._exposure)
        was_running = self._running
        if changed:
            with self.camLock:
                if was_running:
                    self.stopCamera()
                self._region = region
                if 'exposure' in changes:
                    self._exposure = exposure
                self._apply_config()
                if was_running and autoRestart:
                    self.startCamera()
        return {p: self.getParams([p])[p] for p in changes}, bool(changed and was_running)

    def startCamera(self):
        with self.camLock:
            if self._running:
                return
            self._camera.record(number_of_images=10, mode='ring buffer')
            self._camera.wait_for_first_image()
            self._last_recorder_count = None
            self._running = True

    def stopCamera(self):
        with self.camLock:
            if self._running:
                self._camera.stop()
                self._running = False
                self._last_recorder_count = None

    def newFrames(self):
        with self.camLock:
            if not self._running:
                return []
            count = int(self._camera.rec.get_status()['dwProcImgCount'])
            if count == self._last_recorder_count:
                return []
            # Only the newest image is available; previous ring entries may have
            # been overwritten. The recorder count is not a per-image exposure ID.
            image, _metadata = self._camera.image(image_number=self.LATEST_IMAGE)
            data = np.array(image, copy=True)
            retrieved_at = ptime.time()
            previous = self._last_recorder_count
            self._last_recorder_count = count
            # Preserve recorder-detected gaps so ACQ4 reports frames lost from
            # the ring; keep IDs unique across stop/start recorder resets.
            missed = max(0, count - previous - 1) if previous is not None else 0
            frame_id = self._next_frame_id + missed
            self._next_frame_id = frame_id + 1
            return [{'id': frame_id, 'data': data, 'time': retrieved_at,
                     'timeSource': 'hostRetrieval', 'recorderFrameCount': count}]

    def close(self):
        with self.camLock:
            if self._camera is not None:
                self.stopCamera()
                camera, self._camera = self._camera, None
                camera.close()

    def quit(self):
        try:
            super().quit()
        finally:
            self.close()
