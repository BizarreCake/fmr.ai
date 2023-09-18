import contextlib
from typing import Optional, Generator

from fmrai.instrument import instrumentation_scope, get_current_instrumentation_state
from fmrai.logging import log_model, log_model_parameters
from fmrai.tracker import ComputationTracker


class Fmrai:
    def __init__(self):
        self._computation_tracker: Optional[ComputationTracker] = None
        self._models = []

    def add_model(self, model, *, log_parameters=False):
        self._models.append(model)
        log_model(model)
        if log_parameters:
            log_model_parameters(model, time_step=0)

    def _create_computation_tracker(self):
        assert self._computation_tracker is None
        self._computation_tracker = ComputationTracker()

        if len(self._models) == 1:
            self._computation_tracker.set_root_model(self._models[0])

        return self._computation_tracker

    def track_computations(self):
        if self._computation_tracker is None:
            return self._create_computation_tracker()

        self._computation_tracker.reset()
        return self._computation_tracker


@contextlib.contextmanager
def fmrai() -> Generator[Fmrai, None, None]:
    fmr = Fmrai()

    try:
        with instrumentation_scope() as state:
            state.fmr = fmr
            yield fmr
    finally:
        pass


def get_fmrai() -> Fmrai:
    return get_current_instrumentation_state().fmr
