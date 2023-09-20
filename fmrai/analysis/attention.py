import os
import time
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from datasets import Dataset

from pydantic import BaseModel

from fmrai.fmrai import Fmrai
from fmrai.tracker import ComputationMap, TensorId, LazyComputationMap, OrdinalTensorId


class AttentionHeadExtraction(BaseModel):
    matrix: List[List[float]]


class AttentionExtraction(BaseModel):
    heads: List[AttentionHeadExtraction]


def extract_attention_values(
        cmap: ComputationMap,
        tensor_id: TensorId,
) -> List[AttentionExtraction]:
    tensor = cmap.get(tensor_id)

    assert len(tensor.size()) == 4
    batch_size, num_heads, query_len, key_len = tensor.size()

    extractions = []

    for i in range(batch_size):
        heads = []
        for j in range(num_heads):
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
) -> np.ndarray:
    with torch.no_grad():
        # gather all tensors and check that they have the same shape
        all_tensors = []
        for tensor_id in attention_tensors:
            tensor = batch_cmap.get(tensor_id)

            if all_tensors:
                assert (all_tensors[-1].size() == tensor.size())

            all_tensors.append(tensor)

        # concatenate all tensors
        big_tensor = torch.concat(all_tensors, dim=1)  # concatentate along num_heads dimension

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
    x: float
    y: float


class AttentionHeadClusteringResult(BaseModel):
    key: str
    created_at: float
    mds: List[AttentionHeadPoint]
    dataset_name: Optional[str] = None
    limit: Optional[int] = None


def compute_attention_head_clustering(
        cmap: ComputationMap,
        attention_tensors: List[TensorId],
):
    distance_matrix = compute_attention_head_divergence_matrix(
        cmap,
        attention_tensors,
    )

    # apply MDS to get 2d coordinates for each attention head
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_coords = mds.fit_transform((distance_matrix + np.transpose(distance_matrix)) / 2)

    key = os.urandom(8).hex()

    return AttentionHeadClusteringResult(
        key=key,
        created_at=time.time(),
        mds=[
            AttentionHeadPoint(x=row[0], y=row[1])
            for row in mds_coords.tolist()
        ]
    )


def test_extract_attention():
    cmap = LazyComputationMap.load_from('./data/computation_maps/92d9231fb9a645bd')

    print(extract_attention_values(cmap, OrdinalTensorId(ordinal=21)))


if __name__ == '__main__':
    test_extract_attention()
