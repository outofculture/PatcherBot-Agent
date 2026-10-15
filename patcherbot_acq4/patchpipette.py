"""ACQ4 PatchPipette with a private PatcherBot phase map and profile namespace."""
from collections import OrderedDict

from acq4.devices.PatchPipette.patchpipette import PatchPipette
from acq4.devices.PatchPipette.statemanager import PatchPipetteStateManager

from .states import (ApproachCellState, BreakInState, FindPipetteState,
                     GigasealState, HuntCellState)


class PatcherBotStateManager(PatchPipetteStateManager):
    # Never modify ACQ4's shared class-level OrderedDict or global profile map.
    stateHandlers = OrderedDict(PatchPipetteStateManager.stateHandlers)
    stateHandlers.update((state.stateName, state) for state in (
        FindPipetteState, ApproachCellState, HuntCellState, GigasealState, BreakInState,
    ))
    profiles = {}
    _profilesLoadedFromConfig = False

    def __init__(self, dev):
        self._phase_modes = {}
        super().__init__(dev)

    def configureModes(self, mapping: dict[str, str]):
        """Persist phase modes for this pipette's subsequent automatic transitions."""
        phase_names = {'find pipette', 'approach', 'cell detect', 'seal', 'break in'}
        valid = {'Manual', 'Classic', 'Adaptive', 'Agent', 'Training'}
        errors = [(name, mode) for name, mode in mapping.items()
                  if name not in phase_names or mode not in valid]
        if errors:
            raise ValueError(f'Invalid PatcherBot phase mode(s): {errors!r}')
        self._phase_modes = dict(mapping)

    def _configureState(self, state, allowReset=True, configOverride=None):
        if state in self._phase_modes:
            configOverride = {**(configOverride or {}), 'mode': self._phase_modes[state]}
        result = super()._configureState(state, allowReset=allowReset, configOverride=configOverride)
        if state in ('out', 'bath') and getattr(self.dev, 'continuous', None) is not None:
            self.dev.continuous.set_cell_mode(False)
        return result


    @classmethod
    def _loadGlobalProfilesOnce(cls):
        if cls._profilesLoadedFromConfig:
            return
        from acq4 import getManager

        profiles = getManager().config.get('misc', {}).get('patcherbotProfiles', {})
        for name, config in profiles.items():
            cls.addProfile(name, config.copy())
        cls._profilesLoadedFromConfig = True


class PatcherBotPatchPipette(PatchPipette):
    """ACQ4 patch pipette retaining ordinary out/bath/whole-cell/clean states."""

    defaultStateManagerClass = PatcherBotStateManager
