import os
import time
from typing import List, Optional, Iterable

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor
from tqdm import tqdm

from fmrai.analysis.common import DatasetInfo, AnalysisTracker, Analyzer, AnalysisAccumulator
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.fmrai import get_fmrai
from fmrai.instrument import unwrap_proxy
from fmrai.tracker import ComputationMap, TensorId, LazyComputationMap, OrdinalTensorId, BatchedComputationMap, \
    SingleComputationTracker


class AttentionHeadExtraction(BaseModel):
    matrix: List[List[float]]


class AttentionExtraction(BaseModel):
    heads: List[AttentionHeadExtraction]


def extract_attention_values(
        cmap: ComputationMap,
        tensor_id: TensorId,
        head_index: Optional[int] = None,
        instance_range=None,
) -> List[AttentionExtraction]:
    tensor = cmap.get_cat(tensor_id)

    assert len(tensor.size()) == 4
    batch_size, num_heads, query_len, key_len = tensor.size()

    extractions = []

    if instance_range is None:
        instance_range = range(batch_size)

    for i in instance_range:
        if i >= batch_size:
            break

        heads = []
        for j in range(num_heads):
            if head_index is not None and head_index != j:
                continue

            head_tensor = tensor[i, j, :, :]
            heads.append(AttentionHeadExtraction(
                matrix=head_tensor.cpu().tolist(),
            ))

        extractions.append(AttentionExtraction(
            heads=heads,
        ))

    return extractions


def _compute_attention_head_divergence_matrix_one_instance(
        all_heads_tensor: Tensor,
):
    assert len(all_heads_tensor.size()) == 3
    num_heads = all_heads_tensor.size(0)

    # smooth out tensor to prevent issues with logarithm later on
    seq_len = all_heads_tensor.size(1)
    all_heads_tensor = 0.001 / seq_len + all_heads_tensor * 0.999

    js_matrix = np.zeros((num_heads, num_heads))

    for head in range(num_heads):
        head_tensor = all_heads_tensor[head, ...].unsqueeze(0)
        head_tensor = 0.001 / seq_len + head_tensor * 0.999

        # compute jensen-shannon distance between head and all other heads simultaneously

        m = (head_tensor + all_heads_tensor) / 2
        js = -(head_tensor * torch.log2(m / head_tensor) + all_heads_tensor * torch.log2(m / all_heads_tensor)) / 2

        per_head_js = js.sum(dim=-1).sum(dim=-1)
        js_matrix[head] += per_head_js.cpu().numpy()

    return js_matrix / num_heads


def compute_attention_head_divergence_matrix(
        batch_cmap: ComputationMap,
        attention_tensors: List[TensorId],
        *,
        device=None,
) -> np.ndarray:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else None

    with torch.no_grad():
        # gather all tensors and check that they have the same shape
        all_tensors = []
        for tensor_id in attention_tensors:
            tensor = unwrap_proxy(batch_cmap.get_cat(tensor_id))

            if all_tensors:
                assert (all_tensors[-1].size() == tensor.size())

            all_tensors.append(tensor.to(device))

        # concatenate all tensors
        big_tensor = torch.concat(all_tensors, dim=1)  # concatenate along num_heads dimension

        # compute per instance and sum
        instance_count = all_tensors[0].size(0)
        return np.sum(
            np.concatenate(
                [
                    np.expand_dims(_compute_attention_head_divergence_matrix_one_instance(big_tensor[i, ...]), 0)
                    for i in range(instance_count)
                ],
                axis=0,
            ),
            axis=0
        ) / instance_count


class AttentionHeadPoint(BaseModel):
    tensor_id: str
    head_index: int
    x: float
    y: float


class AttentionHeadClusteringResult(BaseModel):
    key: str
    created_at: float
    mds: List[AttentionHeadPoint]
    dataset_info: Optional[DatasetInfo] = None
    limit: Optional[int] = None

    def plot(self, figsize=(8, 8)):
        import seaborn as sns
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            x=[i.x for i in self.mds],
            y=[i.y for i in self.mds],
            hue=[i.tensor_id for i in self.mds],
            ax=ax
        )


class AttentionHeadClusteringAccumulator(AnalysisAccumulator):
    def __init__(self, attention_tensors: List[TensorId]):
        self._attention_tensors = attention_tensors
        self._accumulated_result: Optional[np.ndarray] = None
        self._num_instances = 0

    def process_batch(self, cmap: ComputationMap):
        distance_matrix = compute_attention_head_divergence_matrix(cmap, self._attention_tensors)

        # figure out instance count by looking at batch size in some tensor
        some_tensor_id = next(iter(cmap))
        instance_count = cmap.get(some_tensor_id)[0].size(0)

        # combine with previous batches
        if self._accumulated_result is None:
            self._accumulated_result = distance_matrix / instance_count
        else:
            self._accumulated_result = ((self._accumulated_result * self._num_instances + distance_matrix) /
                                        (self._num_instances + instance_count))

        self._num_instances += instance_count

    def result(self) -> Optional[np.ndarray]:
        return self._accumulated_result


def _compute_attention_plot_coords_from_distance_matrix(
        distance_matrix: np.ndarray,
        attention_tensors: List[TensorId],
        ref_batch,  # used to compute the number of attention heads
):
    """
    Apply MDS to get 2d coordinates for each attention head.
    """
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_coords = mds.fit_transform((distance_matrix + np.transpose(distance_matrix)) / 2)

    key = os.urandom(8).hex()

    heads = []
    tensor_index = -1
    head_index = 0
    heads_in_tensor = 0
    for i, row in enumerate(mds_coords.tolist()):
        if heads_in_tensor == 0:
            head_index = 0
            tensor_index += 1
            heads_in_tensor = ref_batch.get(attention_tensors[tensor_index])[0].size(1)

        heads.append(AttentionHeadPoint(
            tensor_id=str(attention_tensors[tensor_index]),
            head_index=head_index,
            x=row[0],
            y=row[1],
        ))

        head_index += 1
        heads_in_tensor -= 1

    return AttentionHeadClusteringResult(
        key=key,
        created_at=time.time(),
        mds=heads,
    )


def compute_attention_head_clustering(
        cmap: ComputationMap,
        attention_tensors: List[TensorId],
) -> AttentionHeadClusteringResult:
    if isinstance(cmap, BatchedComputationMap):
        batches = list(cmap)
    else:
        batches = [cmap]

    # process all batches
    batches_iter = tqdm(batches, desc='computing attention head divergence matrix')
    accumulator = AttentionHeadClusteringAccumulator(attention_tensors)
    for batch in batches_iter:
        accumulator.process_batch(batch)

    distance_matrix = accumulator.result()
    return _compute_attention_plot_coords_from_distance_matrix(
        distance_matrix,
        attention_tensors,
        batches[0],
    )


class AttentionTracker(AnalysisTracker):
    def __init__(self):
        super().__init__()
        self._attention_tensor_ids: Optional[List[TensorId]] = None
        self._expected_cmap_size: Optional[int] = None

    def _process_batch(self, cmap: ComputationMap, tracker: SingleComputationTracker):
        if self._attention_tensor_ids is None:
            self._find_attention_tensors(tracker)

        if self._expected_cmap_size is not None:
            if self._expected_cmap_size != tracker.num_seen_tensors:
                raise Exception(f'Expected {self._expected_cmap_size} tensors, got {tracker.num_seen_tensors} tensors.')
        else:
            self._expected_cmap_size = tracker.num_seen_tensors
            filter_set = set(self._attention_tensor_ids)
            cmap = cmap.filter_ids(lambda x: x in filter_set)

        return cmap

    def _get_tracked_tensors(self) -> Optional[Iterable[TensorId]]:
        return self.attention_tensor_ids

    @property
    def attention_tensor_ids(self) -> Optional[List[TensorId]]:
        return self._attention_tensor_ids

    def _find_attention_tensors(self, tracker: SingleComputationTracker):
        """
        Called after processing the first batch to find the ids of the attention tensors.
        """
        cg = tracker.build_graph()
        result = list(find_multi_head_attention(cg))

        self._attention_tensor_ids = [
            instance.softmax_value.tensor_id
            for instance in result
        ]


class AttentionHeadClusterAnalyzer(Analyzer):
    def _create_tracker(self) -> AnalysisTracker:
        return AttentionTracker()

    def _create_accumulator(self) -> AnalysisAccumulator:
        return AttentionHeadClusteringAccumulator(self.tracker.attention_tensor_ids)

    def analyze(self):
        """ Analyzes all available output produced by the tracker(s). """
        with get_fmrai().pause():
            # consume remaining batches
            while self.tracker:
                self._consume_batch_from_tracker()

            distance_matrix = self._accumulator.result()

            assert self.tracker.attention_tensor_ids is not None
            result = _compute_attention_plot_coords_from_distance_matrix(
                distance_matrix,
                self.tracker.attention_tensor_ids,
                ref_batch=self._first_batch,
            )

        return result


def test_extract_attention():
    cmap = LazyComputationMap.load_from('./data/computation_maps/92d9231fb9a645bd')

    print(extract_attention_values(cmap, OrdinalTensorId(ordinal=21)))


if __name__ == '__main__':
    test_extract_attention()
