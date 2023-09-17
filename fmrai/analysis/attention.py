from typing import List

from pydantic import BaseModel

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


def test_extract_attention():
    cmap = LazyComputationMap.load_from('./data/computation_maps/92d9231fb9a645bd')

    print(extract_attention_values(cmap, OrdinalTensorId(ordinal=21)))


if __name__ == '__main__':
    test_extract_attention()
