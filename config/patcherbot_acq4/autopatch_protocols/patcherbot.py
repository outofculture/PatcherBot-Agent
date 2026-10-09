"""Run PatcherBot phase states from ACQ4's Autopatch experiment orchestrator.

Set ``protocolModes`` in the PatcherBot module config or pass ``modes`` to run.
The module owns one attempt recorder and optional continuous DAQ stream; the
ACQ4 patch action owns the cell queue, state transitions and terminal outcome.
"""

from acq4.experiment.actions import patch
from patcherbot_acq4.patchpipette import PatcherBotPatchPipette


def run(ctx, *, modes=None):
    if not isinstance(ctx.pipette, PatcherBotPatchPipette):
        raise TypeError("This protocol requires a PatcherBotPatchPipette")
    module = ctx.manager.getOrLoadModule("PatcherBot")
    if module.pipette is not ctx.pipette:
        raise RuntimeError("PatcherBot module and Autopatch target different pipettes")
    selected = modes if modes is not None else module.config.get("protocolModes")
    if selected is None:
        raise ValueError("Specify protocolModes in the PatcherBot module config before running Autopatch")
    module.prepareAttempt("approach", selected)
    try:
        # The extension already records ACQ4 events and full test pulses.
        outcome = patch(ctx, record_events=False)
        ctx.log(f"PatcherBot phase outcome: {outcome}")
        return outcome
    except BaseException:
        module.abortAttempt()
        raise
    finally:
        module.finishAttempt()
