# PatcherBot-Agent


This code has its basis in [HolyPipette](https://github.com/romainbrette/holypipette). Many 
Thanks to the original contributors.

See the [documentation](docs/index.rst) for installation and usage.

## Conda setup and install order
1. `conda create -n patcherbot python=3.10.11`
2. `conda activate patcherbot`
3. `pip install -r requirements/requirements_runtime.txt` (core runtime deps)
4. Obtain the author's custom robomimic fork (required for standalone Agent modes).
5. `pip install -e /path/to/custom/robomimic`
6. `pip install -e .`

Notes:
- SAM2 segmentation and LightGlue matching are loaded via Hugging Face Transformers (`transformers>=4.56.0`; models download on first use).

## ACQ4 extension

The experimental `patcherbot_acq4` package runs PatcherBot's five phase jobs
inside ACQ4's PatchPipette state manager. Its separate, simulated rig is
`config/patcherbot_acq4/default.cfg`; installation, hardware configurations,
recording and missing-model limitations are documented in
[`config/patcherbot_acq4/README.md`](config/patcherbot_acq4/README.md).
Install into an existing ACQ4 environment with `python -m pip install -e . --no-deps`.
The legacy standalone application remains available until physical rig and
policy parity can be validated; it is not a fallback for the ACQ4 extension.
