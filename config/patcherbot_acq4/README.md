# PatcherBot inside ACQ4

Install this repository in the ACQ4 Python environment with `python -m pip install -e . --no-deps` (the legacy PatcherBot application's pinned dependencies conflict with ACQ4). Then run `QT_QPA_PLATFORM=offscreen python -m acq4 -x -c config/patcherbot_acq4/default.cfg -m Camera -m PatcherBot` for a headless load check, or omit `QT_QPA_PLATFORM` for the interactive window. Create/select an ACQ4 storage directory before beginning an attempt. `default.cfg` is a self-contained simulated ACQ4 rig; it does not validate real actuators or reproduce a PatcherBot experiment.

Model inference requires the `acq4-detectors` optional dependencies listed in
`setup.py`; the continuous service needs `nidaqmx` plus the installed NI driver,
and a physical PCO camera needs the vendor `pco` SDK. Install only the needed
optional packages in the ACQ4 environment, independently of the legacy
standalone application's pinned requirements.

The PatcherBot window starts disarmed. Configure a mode separately for find pipette, approach, cell detect (PatcherBot hunt), seal (gigaseal) and break in. Arm explicitly to start. Abort and vent transitions to ACQ4's safe `out` state and vents the pressure; the pipette stays ACQ4-active until `out` finishes initializing, so deactivate it from MultiPatch after it settles if desired. Training success/abort is an operator decision, not a synthetic model action. ACQ4 still owns device reservations, patch logs and motion planning.

Before Classic/Adaptive geometry moves, set an ACQ4 pipette target and mark the
sample surface on its microscope. The motion planner refuses moves without a
surface depth or beyond manipulator limits; it does not fall back to direct
unplanned movement. The mock config does not invent a sample surface or target.

`patchProfiles.cfg` is a **private** PatcherBotPatchPipette profile, not a change
to ACQ4's stock PatchPipette states. Values are editable through ACQ4's profile
API. The UI mode selectors override the profile for one attempt. For ACQ4
Autopatch, select `config/patcherbot_acq4/autopatch_protocols/patcherbot.py` and
explicitly configure all five `protocolModes` in the PatcherBot module config
(the commented block in `default.cfg` is an example). That protocol attaches
the same attempt recorder/NI service, then delegates state progression to
`acq4.experiment.actions.patch`; it does not run a second automation queue.
Do not run an Autopatch protocol and a manual PatcherBot attempt concurrently.

## Model artifacts

The tip/defocus inference code is carried from `../Neuron_Detection/code/Consumers/PipetteModel/PDModelFactory.py` (MIT; copy at `patcherbot_acq4/model_runtime/`). It implements a 384 px grayscale+Sobel U-Net/MobileNetV4 model, *not* ACQ4's ResNet pipette tracker. Specify its canonical checkpoint as `detector.pipetteCheckpoint`; Adaptive visual approach/hunt also requires `detector.cellCheckpoint` (and `cellEncoderConfig` for SegFormer). There is no checkpoint bundled. Missing or incompatible checkpoints raise before visual control begins.

ACQ4/PCO cameras may supply raw `uint16` frames; PatcherBot's original camera
normalizes to `uint8` before inference. Set `detector.inputRange: [lower, upper]`
from measured sensor intensities to map raw counts to 0–255; the raw frame is
still recorded separately. Without this explicit calibration, 16-bit
visual/Agent inference fails rather than wrapping modulo 256. Check that
lighting and this normalization match the detector/policy training domain.

Agent modes require explicit PipetteFinder/GigaSealer/Burglar `.pth` checkpoints and the **custom robomimic fork** containing `robomimic.envs.env_patcher_online.EnvPatcherOnline`. Specify `policy.robomimicRoot` as the parent of the `robomimic/` package and `policy.checkpoints` by phase key `find_pipette`, `gigaseal`, `break_in`. PatcherBot's source hunt Agent has no movement policy action, and no approach Agent policy is deployed; both are blocked. Find-pipette velocity-prediction mode is blocked until a calibrated continuous-velocity stage API is available; displacement mode uses bounded ACQ4 planned moves. No weights, fork or matched labelled demonstration logs were in these checkouts. `rapiddweller/datamimic` is unrelated synthetic enterprise data software.

## Continuous membrane test

The optional `continuous` section uses real NI-DAQmx tasks: 50 kS/s AI, AI-triggered regenerating AO, 40 Hz / 10 mV command, timestamped most-recent periods. Set explicit physical AI/AO channels, trigger terminal and measured MultiClamp gains. It exclusively reserves the ACQ4 NiDAQ while active; pause before other finite DAQ protocols. It never runs on the mock NiDAQ. Bath mode reports total resistance; sealing/break-in switch to the cell-mode Ra/Rm/Cm fit and invalidate older samples. `ContinuousMembraneTest.latest(max_age=...)` rejects missing/stale/invalid samples. Seal and break-in Agent states require this stream rather than silently consuming ACQ4's different finite test pulses; find-pipette displacement Agent uses images and needs no stream. **No physical DAQ has been tested by this config.** Bench-verify command amplitude, polarity, gain and zero-output on stop before connecting a cell.

## Driver configuration

All extension classes register through `imports: ['patcherbot_acq4']` and use ACQ4's `(manager, config, name)` constructor. Add these entries to a real rig's `devices.cfg` after checking pinout and calibration; the example values below are *not* executable rig calibrations:

```text
Pressure1: driver 'PatcherBotSerialPressure', variant 'Moscow'/'IBB'/'Waynesboro', channel, port, readerPort (optional), minimum/maximum in Pa, nativeZero/nativePerMbar and readerOffset/readerScale if measured. `getPressure()` is the setpoint in Pa; `measurePressure()` queries an independent sensor, and errors if absent. Panic vents.
Camera: driver 'PatcherBotPcoCamera', parentDevice 'Microscope', sensorSize (width,height), region (x,y,width,height) zero-based pixels, exposure seconds. Frame time is host retrieval, not exposure. SDK ring frames may be dropped.
Olympus: driver 'PatcherBotOlympusLamp', port; cubeSlots and cubeCommand (CUBE/MU) optional.
Lumencor: driver 'PatcherBotLumencor', port; wraps ACQ4's SpectraXDriver.
Excelitas: driver 'PatcherBotExcelitasWheel', daqDevice, stepChannel, directionChannel, slots and initialSlot (physical reference). Position becomes unknown after an interrupted move.
Encoder stage: driver 'PatcherBotEncoderStage', port, encoderPort, stageUnitsPerEncoderPulse (calibrated µm/count). Two provided encoder firmware variants conflict (2.178649 vs 1.45 µm/count); there is no safe default.
```

**CellSorter is blocked:** the PatcherBot repo ignores and omits `patcherbot/devices/cellsorter/CellSorterSerial.py`, which defines the proprietary serial opcodes for valves, LEDs and stage motion. No controller/manipulator driver can be implemented safely without that source or a protocol specification. Request it from the author; never infer opcodes from symbolic names.

## Evidence and limitations

Each attempt writes `phase_events.jsonl` with phase observations, calculations, actions, original sensor clocks, and paired monotonic/wall-clock ACQ4 device events; image/array payloads are stored as `.npy`. `manifest.json` records model SHA-256 hashes and clock anchor. The ACQ4 MultiPatch log and full-test-pulse sidecar preserve the raw electrophysiology when test pulses exist. Training mode records an operator success/abort label and the raw motion/pressure events rather than inventing an action vector. **No verified conversion to the existing per-phase robomimic HDF5 schema is possible yet:** action alignment, anchoring, and preprocessing require the author's timestamped demonstrations, checkpoint metadata and rig calibration. The included PatcherBot `temp_dataset.hdf5` has two short demos, uint8 resistance and no per-sample timestamps or matched live actions. ACQ4 has real recorded seal traces under `devices/PatchPipette/tests/data/` for threshold tests, not PatcherBot training data. Do not claim learned-mode or real-rig performance from the mock run.
