import functools
import operator
from enum import IntEnum, auto
from typing import Optional, List, Iterable, Set

import torch

from fmrai.analysis.common import AnalysisTracker, Analyzer, AnalysisAccumulator
from fmrai.analysis.structure import FindTransformerFFN, TransformerFFNInstance
from fmrai.tracker import ComputationMap, SingleComputationTracker, TensorId


class KeyValueMaxSearchStrategy(IntEnum):
    PREFIX = auto()
    """ Inspects each sentence prefix when searching for maximum memory coefficients. """

    CLS = auto()
    """ Inspects only the CLS (first token) when searching for maximum memory coefficients. """


class KeyValueAnalysisAccumulator(AnalysisAccumulator):
    def __init__(
            self,
            ffns: List[TransformerFFNInstance],
            strategy: KeyValueMaxSearchStrategy,
            max_entries: int = 10,
    ):
        super().__init__()
        self._ffns = ffns
        self.strategy = strategy
        self.max_entries = max_entries

        self._mem_coeff_bank = {}
        self._instance_idx = 0

    def process_batch(self, cmap: ComputationMap):
        with torch.no_grad():
            for layer_idx, ffn in enumerate(self._ffns):
                tensors = cmap.get(ffn.act.tensor_id)
                assert len(tensors) == 1
                tensor = tensors[0]

                if self.strategy == KeyValueMaxSearchStrategy.CLS:
                    cls_only = tensor[:, 0, :].numpy()

                    # go over all sentences in batch
                    for i in range(cls_only.shape[0]):
                        mem_coeffs = cls_only[i]

                        for j in range(mem_coeffs.shape[0]):
                            key = (layer_idx, j)
                            value = (self._instance_idx, mem_coeffs[j].item())

                            if key not in self._mem_coeff_bank:
                                self._mem_coeff_bank[key] = [value]
                            else:
                                # add, re-sort, and keep best entries
                                self._mem_coeff_bank[key] = sorted(
                                    self._mem_coeff_bank[key] + [value],
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:self.max_entries]

                        self._instance_idx += 1

    def result(self):
        pass


class KeyValueAnalysisTracker(AnalysisTracker):
    def __init__(self):
        super().__init__()

        self._ffns: Optional[List[TransformerFFNInstance]] = None
        self._relevant_ids: Optional[Set[TensorId]] = None

    @property
    def ffns(self) -> Optional[List[TransformerFFNInstance]]:
        return self._ffns

    def _get_tracked_tensors(self) -> Optional[Iterable[TensorId]]:
        return self._relevant_ids

    def _process_batch(self, cmap: ComputationMap, tracker: SingleComputationTracker) -> ComputationMap:
        if self._relevant_ids is None:
            cg = tracker.build_graph()
            self._ffns = list(FindTransformerFFN(cg.g).search())

            self._relevant_ids = functools.reduce(
                operator.or_,
                ({ffn.act.tensor_id, ffn.linear_bottom.tensor_id} for ffn in self._ffns),
                set(),
            )

        cmap = cmap.filter_ids(lambda x: x in self._relevant_ids)
        return cmap


class KeyValueAnalyzer(Analyzer):
    def __init__(self, strategy: KeyValueMaxSearchStrategy):
        super().__init__()
        self.strategy = strategy

    def _create_tracker(self) -> AnalysisTracker:
        return KeyValueAnalysisTracker()

    def _create_accumulator(self) -> AnalysisAccumulator:
        assert self._tracker is not None
        assert isinstance(self._tracker, KeyValueAnalysisTracker)
        return KeyValueAnalysisAccumulator(
            ffns=self._tracker.ffns,
            strategy=self.strategy
        )

    def analyze(self):
        pass
