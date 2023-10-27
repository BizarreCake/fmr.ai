import contextlib
from typing import Optional, Generator, Union, Iterable

from fmrai.instrument import instrumentation_scope, get_current_instrumentation_state, pause_instrumentation
from fmrai.logging import log_model, log_model_parameters
from fmrai.tracker import SingleComputationTracker, BatchedComputationTracker, TensorId


class Fmrai:
    def __init__(self):
        self._computation_tracker: Optional[SingleComputationTracker] = None
        self._models = []

    def add_model(self, model, *, log_parameters=False):
        self._models.append(model)
        log_model(model)
        if log_parameters:
            log_model_parameters(model, time_step=0)

    def track(
            self,
            *,
            batched=False,
            track_tensors: Optional[Iterable[TensorId]] = None,
    ) -> Union[SingleComputationTracker, BatchedComputationTracker]:
        if batched:
            tracker = BatchedComputationTracker(track_tensors=track_tensors)
        else:
            tracker = SingleComputationTracker(track_tensors=track_tensors)

        if len(self._models) == 1:
            tracker.set_root_model(self._models[0])

        return tracker

    @contextlib.contextmanager
    def pause(self):
        with pause_instrumentation():
            yield


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
