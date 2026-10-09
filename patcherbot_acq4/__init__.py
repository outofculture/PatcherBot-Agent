"""PatcherBot's ACQ4 module and hardware extensions.

ACQ4 discovers Module and Device subclasses after this package is imported by
its `imports:` configuration key. Heavy model and vendor SDKs load on demand.
"""

from .module import PatcherBotModule
from .patchpipette import PatcherBotPatchPipette
from . import devices as _devices

__all__ = ["PatcherBotModule", "PatcherBotPatchPipette"]
