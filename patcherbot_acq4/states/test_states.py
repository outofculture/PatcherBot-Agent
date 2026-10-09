"""PatcherBot phase gates and ACQ4 isolation (no hardware connected)."""
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from patcherbot_acq4.patchpipette import PatcherBotStateManager
from patcherbot_acq4.states import BreakInState, GigasealState, HuntCellState, PhaseContext
from acq4.devices.PatchPipette.statemanager import PatchPipetteStateManager


def test_private_state_registry_and_automatic_transitions():
    manager = SimpleNamespace(_phase_modes={})
    PatcherBotStateManager.configureModes(manager, {'cell detect': 'Adaptive', 'seal': 'Training'})
    assert manager._phase_modes == {'cell detect': 'Adaptive', 'seal': 'Training'}
    assert PatcherBotStateManager.stateHandlers['cell detect'] is HuntCellState
    assert PatcherBotStateManager.stateHandlers['seal'] is GigasealState
    assert PatchPipetteStateManager.stateHandlers['cell detect'] is not HuntCellState
    with pytest.raises(ValueError, match='Invalid PatcherBot phase mode'):
        PatcherBotStateManager.configureModes(manager, {'cell detect': 'unknown'})


def test_hunt_requires_five_readings_and_fresh_adaptive_confirmation():
    job = SimpleNamespace(config={'mode': 'Classic', 'cellRiseMohm': .3}, ctx=None)
    state = {'baseline': 5., 'readings': deque(maxlen=5), 'start': np.zeros(3)}
    results = []
    for resistance in (5., 5.1, 5.2, 5.3, 5.31):
        results.append(HuntCellState.calculate(job, {'resistance': resistance, 'position_um': np.zeros(3)}, state))
    assert [sample['resistance_up'] for sample in results] == [False, False, False, False, True]
    assert results[-1]['resistance_delta_mohm'] == pytest.approx(.31)


def test_break_in_three_low_access_measurements():
    job = SimpleNamespace(config={'mode': 'Classic', 'maxAccessMohm': 70.})
    state = {'good': 0}
    for access in (69., 71., 69., 69., 69.):
        action = BreakInState.decide(job, {'observation': {'access_resistance': access},
                                           'trial': 1, 'wait_s': 1., 'pulse_duration_s': 1.}, state)
    assert action == {'kind': 'complete'}
    assert state['good'] == 3


def test_recorded_acq4_seal_trace_uses_mohm_boundary():
    # Actual ACQ4 PatchPipette saved resistance trace (time, Ohms, membrane current).
    fixture = (Path(__file__).resolve().parents[2] / '..' / 'acq4' / 'acq4' / 'devices'
               / 'PatchPipette' / 'tests' / 'data' / 'spontaneous_break_in_seal.npy')
    recorded = np.load(fixture)
    resistance_mohm = recorded[:, 1] / 1e6
    assert resistance_mohm[:5].mean() < 25
    assert resistance_mohm.max() >= 1000
    jumps = np.where(np.diff(resistance_mohm) > 15)[0]
    assert jumps.size
    index = int(jumps[0])
    state = {'previous': resistance_mohm[index], 'last_progress': 0.}
    job = SimpleNamespace(config={'minDeltaMohm': 15., 'measurementIntervalS': .2})
    calc = GigasealState.calculate(job, {'resistance': resistance_mohm[index + 1]}, state)
    assert calc['rate_mohm_s'] == pytest.approx(
        (resistance_mohm[index + 1] - resistance_mohm[index]) / (5 * .2))
    assert state['last_progress'] > 0


def test_agent_preflight_rejects_missing_policy_before_actuation():
    pip = SimpleNamespace(target=np.array([0., 0., 0.]))
    dev = SimpleNamespace(pipetteDevice=pip, clampDevice=object(), pressureDevice=object(),
                          camera=object(), detector=None, policy=None, recorder=None, continuous=None)
    job = SimpleNamespace(dev=dev, config={'mode': 'Agent'}, stateName='seal', policy_phase='gigaseal')
    with pytest.raises(RuntimeError, match='dev.policy'):
        PhaseContext(job).preflight()


def test_planned_motion_is_bounded_and_refuses_stale_observation():
    class Pipette:
        def globalPosition(self):
            return np.zeros(3)

        def moveToGlobalNoPlanning(self, *args, **kwargs):
            raise AssertionError('Direct unplanned move must never be used')

    moves = []
    manager = SimpleNamespace(move=lambda spec, **kwargs:
                              (moves.append(spec), SimpleNamespace(wait=lambda: None))[1])
    dev = SimpleNamespace(pipetteDevice=Pipette(), clampDevice=None, pressureDevice=None,
                          camera=None, detector=None, policy=None, recorder=None, dm=manager)
    job = SimpleNamespace(dev=dev, config={'maxMoveStepUm': 100.}, stateName='find pipette')
    ctx = PhaseContext(job)
    ctx.last_observed_at = time.monotonic()
    assert ctx.move_um((300., 0., 0.), 200.) is False
    assert len(moves) == 1
    assert moves[0].position == pytest.approx([100e-6, 0., 0.])
    ctx.last_observed_at = time.monotonic() - 5
    with pytest.raises(RuntimeError, match='stale'):
        ctx.move_um((1., 0., 0.), 200.)
    assert len(moves) == 1
