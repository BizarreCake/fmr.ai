import collections
import contextlib
import functools
import operator
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List, Iterable, Set, Deque, Tuple, Dict

import torch
from torch import Tensor

from fmrai.analysis.common import AnalysisTracker, Analyzer, AnalysisAccumulator, Batch
from fmrai.analysis.structure import FindTransformerFFN, TransformerFFNInstance
from fmrai.fmrai import get_fmrai
from fmrai.instrument import unwrap_proxy
from fmrai.tracker import ComputationMap, SingleComputationTracker, TensorId


class KeyValueMaxSearchStrategy(IntEnum):
    PREFIX = auto()
    """ Inspects each sentence prefix when searching for maximum memory coefficients. """

    CLS = auto()
    """ Inspects only the CLS (first token) when searching for maximum memory coefficients. """


@dataclass
class KVBatch(Batch):
    attention_mask: Optional[Tensor] = None


KVBankKey = Tuple[int, int]
KVBankValueEntry = Tuple[int, int, float]
KVBankData = Dict[KVBankKey, List[KVBankValueEntry]]

KVBankAverages = Dict[KVBankKey, float]


class KVResultBank:
    def __init__(self, data: KVBankData):
        self.data = data


class KeyValueAnalysisAccumulator(AnalysisAccumulator):
    def __init__(
            self,
            ffns: List[TransformerFFNInstance],
            strategy: KeyValueMaxSearchStrategy,
            start_index: int = 0,
            max_entries: int = 10,
    ):
        super().__init__()
        self._ffns = ffns
        self.strategy = strategy
        self.max_entries = max_entries

        self._kv_bank_data = {}

        self._instance_idx = start_index

    def process_batch(self, batch: KVBatch):
        cmap = batch.cmap

        batch_start_idx = self._instance_idx
        batch_size = 0

        with torch.no_grad():
            for layer_idx, ffn in enumerate(self._ffns):
                tensors = cmap.get(ffn.act.tensor_id)
                assert len(tensors) == 1
                tensor = tensors[0]

                batch_size = tensor.size(0)

                if self.strategy == KeyValueMaxSearchStrategy.CLS:
                    #
                    # look only at the CLS token
                    #
                    cls_only = tensor[:, 0, :].numpy()

                    # go over all sentences in batch
                    for batch_index in range(cls_only.shape[0]):
                        instance_idx = batch_start_idx + batch_index
                        mem_coeffs = cls_only[batch_index]

                        for prefix_index in range(mem_coeffs.shape[0]):
                            key = (layer_idx, prefix_index)
                            value = (instance_idx, mem_coeffs[prefix_index].item())

                            if key not in self._kv_bank_data:
                                self._kv_bank_data[key] = [value]
                            else:
                                # add, re-sort, and keep best entries
                                self._kv_bank_data[key] = sorted(
                                    self._kv_bank_data[key] + [value],
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:self.max_entries]

                elif self.strategy == KeyValueMaxSearchStrategy.PREFIX:
                    #
                    # consider all sentence prefixes
                    #

                    # go over batch
                    for batch_index in range(tensor.size(0)):
                        instance_idx = batch_start_idx + batch_index

                        # determine sequence length from attention mask
                        seq_len = tensor.size(1)
                        if batch.attention_mask is not None:
                            seq_len = batch.attention_mask[batch_index].sum().item()

                        truncated_instance = tensor[batch_index, :seq_len, :]
                        max_result = torch.max(truncated_instance, dim=0)

                        for neuron_index, (prefix_index, mem_coeff), in enumerate(zip(max_result.indices, max_result.values)):
                            key = (layer_idx, neuron_index)
                            value = (
                                instance_idx,
                                prefix_index.item(),
                                mem_coeff.item(),
                            )

                            if key not in self._kv_bank_data:
                                self._kv_bank_data[key] = [value]
                            else:
                                # add, re-sort, and keep best entries
                                self._kv_bank_data[key] = sorted(
                                    self._kv_bank_data[key] + [value],
                                    key=lambda x: abs(x[2]),
                                    reverse=True
                                )[:self.max_entries]

        self._instance_idx += batch_size

    def result(self) -> KVResultBank:
        return KVResultBank(self._kv_bank_data)


class KeyValueAnalysisTracker(AnalysisTracker):
    def __init__(self):
        super().__init__()

        self._ffns: Optional[List[TransformerFFNInstance]] = None
        self._relevant_ids: Optional[Set[TensorId]] = None

        self._attention_masks: Deque[Tensor] = collections.deque()

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

    def track_batch(
            self,
            *,
            attention_mask: Optional[Tensor] = None,
    ):
        with get_fmrai().pause():
            self._attention_masks.append(
                unwrap_proxy(attention_mask).cpu().detach() if attention_mask is not None else None
            )

        return super().track_batch()

    def consume_batch(self) -> KVBatch:
        """ Pops the next batch. """
        return KVBatch(
            cmap=self._cmaps.popleft(),
            attention_mask=self._attention_masks.popleft(),
        )


class KeyValueAnalyzer(Analyzer):
    def __init__(self, strategy: KeyValueMaxSearchStrategy):
        super().__init__()
        self.strategy = strategy

        self._start_index = 0

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        """ Sets the index of the first instance."""
        self._start_index = value

    def _create_tracker(self) -> AnalysisTracker:
        return KeyValueAnalysisTracker()

    def _create_accumulator(self) -> AnalysisAccumulator:
        assert self._tracker is not None
        assert isinstance(self._tracker, KeyValueAnalysisTracker)
        return KeyValueAnalysisAccumulator(
            ffns=self._tracker.ffns,
            strategy=self.strategy,
            start_index=self._start_index,
        )

    def analyze(self):
        pass
