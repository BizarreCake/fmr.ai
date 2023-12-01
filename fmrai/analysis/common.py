import collections
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Deque, Iterable, List, Dict

import networkx as nx
from pydantic import BaseModel

from fmrai.fmrai import get_fmrai
from fmrai.tracker import ComputationMap, SingleComputationTracker, TensorId


@dataclass
class Batch:
    cmap: ComputationMap


class AnalysisAccumulator(ABC):
    @abstractmethod
    def process_batch(self, batch: Batch):
        raise NotImplementedError()

    @abstractmethod
    def result(self):
        raise NotImplementedError()


class AnalysisTracker:
    def __init__(self):
        self._cmaps: Deque[ComputationMap] = collections.deque()

    def _process_batch(self, cmap: ComputationMap, tracker: SingleComputationTracker) -> ComputationMap:
        """
        Processes a batch before it is stored.
        Returns the computation map that should be stored.
        """
        return cmap

    def _get_tracked_tensors(self) -> Optional[Iterable[TensorId]]:
        """
        Returns a list of tensors that should be included in resulting computation maps.
        If None is returned, tracks all tensors.
        """
        return None

    @contextlib.contextmanager
    def track_batch(self, *args, **kwargs):
        """
        Returns a context manager that will process a single batch within its scope.
        """
        fmr = get_fmrai()

        with fmr.track(track_tensors=self._get_tracked_tensors()) as tracker:
            tracker: SingleComputationTracker

            # pass control back and wait for computation
            yield

            cmap = tracker.build_map()
            cmap = self._process_batch(cmap, tracker)
            assert isinstance(cmap, ComputationMap)
            self._cmaps.append(cmap)

    def __bool__(self):
        """ Returns true if consume_batch() has any output."""
        return len(self._cmaps) > 0

    def consume_batch(self) -> Batch:
        """ Pops the next batch. """
        return Batch(
            cmap=self._cmaps.popleft(),
        )


class Analyzer(ABC):
    def __init__(
            self,
            tracker: Optional[AnalysisTracker] = None,
    ):
        self._tracker = tracker
        self._accumulator: Optional[AnalysisAccumulator] = None
        self._first_batch: Optional[Batch] = None

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = self._create_tracker()
        return self._tracker

    def _consume_batch_from_tracker(self):
        with get_fmrai().pause():
            batch = self.tracker.consume_batch()
            if self._first_batch is None:
                self._first_batch = batch

            self._accumulator.process_batch(batch)

    @contextlib.contextmanager
    def track_batch(self, *args, **kwargs):
        with self.tracker.track_batch(*args, **kwargs):
            yield

        if self._accumulator is None:
            self._accumulator = self._create_accumulator()

        self._consume_batch_from_tracker()

    @abstractmethod
    def _create_tracker(self) -> AnalysisTracker:
        raise NotImplementedError()

    @abstractmethod
    def _create_accumulator(self) -> AnalysisAccumulator:
        raise NotImplementedError()

    @abstractmethod
    def analyze(self):
        """ Analyzes all available output produced by the tracker(s). """
        raise NotImplementedError()


def weak_topological_sort(g: nx.DiGraph, nodes):
    if not isinstance(nodes, set):
        nodes = set(nodes)

    sccs = list(nx.strongly_connected_components(g))
    condensed = nx.condensation(g, scc=sccs)

    result = []
    for scc_index in nx.topological_sort(condensed):
        scc = sccs[scc_index]
        for node in scc:
            if node in nodes:
                result.append(node)

    return result


class DatasetInfo(BaseModel):
    name: str
    text_column: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DatapointTokenization:
    words: List[str]


class TokenizationHelper(ABC):
    def tokenize(self, index: int) -> DatapointTokenization:
        raise NotImplementedError()

    def tokenize_many(self, indices: Iterable[int]) -> Dict[int, DatapointTokenization]:
        if not isinstance(indices, set):
            indices = set(indices)
        return {index: self.tokenize(index) for index in indices}


class HFTokenizationHelper(TokenizationHelper):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def tokenize(self, index: int) -> DatapointTokenization:
        datapoint = self.dataset[index]

        # filter out padding tokens using attention mask
        input_ids = datapoint['input_ids']
        attention_mask = datapoint['attention_mask']
        input_ids = [input_ids[i] for i in range(len(input_ids)) if attention_mask[i] == 1]

        words = self.tokenizer.convert_ids_to_tokens(input_ids)
        return DatapointTokenization(words=words)
