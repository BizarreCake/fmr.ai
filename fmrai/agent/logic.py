import json
import os
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel
import numpy as np

import torch

from fmrai.agent import AgentState
from fmrai.agent.api import TokenizedText
from fmrai.analysis.attention import compute_attention_head_divergence_matrix, compute_attention_head_clustering
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.fmrai import get_fmrai
from fmrai.logging import get_log_dir, get_attention_head_plots_dir


@dataclass
class TextPredictionResult:
    activation_map_key: str
    result: TokenizedText


def do_init_model(agent_state: AgentState):
    fmr = get_fmrai()

    with fmr.track_computations() as tracker:
        output = agent_state.api.predict_zero()
        graph = tracker.build_graph(output)

    nice_graph = graph.make_nice()
    nice_graph.save(get_log_dir(), 'computation_graph', save_dot=True)


def do_predict_text(agent_state: AgentState, text: str) -> TextPredictionResult:
    fmr = get_fmrai()

    with fmr.track_computations() as tracker:
        with torch.no_grad():
            result = agent_state.api.predict_text_one(text)
        mp = tracker.build_map()
        print('mp', len(mp.data))

    map_key = os.urandom(8).hex()
    mp.save(map_key)

    return TextPredictionResult(
        activation_map_key=map_key,
        result=result,
    )


class AttentionHeadPoint(BaseModel):
    x: float
    y: float


class AttentionHeadClusteringResult(BaseModel):
    dataset_name: str
    limit: Optional[int]
    mds: List[AttentionHeadPoint]


def do_compute_attention_head_plot(agent_state: AgentState, dataset_name: str, limit: int):
    ds, ds_info = agent_state.api.load_dataset(dataset_name)

    fmr = get_fmrai()

    # find attention heads first
    with fmr.track_computations() as tracker:
        y = agent_state.api.predict_zero()
        g = tracker.build_graph(y).make_nice()

    heads = list(find_multi_head_attention(g))
    attention_tensor_ids = [h.softmax_value.tensor_id for h in heads]

    with fmr.track_computations() as tracker:
        with torch.no_grad():
            agent_state.api.predict_text_many(ds, ds_info.text_column, limit=limit)
        mp = tracker.build_map(tensors=attention_tensor_ids)

    result = compute_attention_head_clustering(mp, attention_tensor_ids)
    result.dataset_name = dataset_name
    result.limit = limit

    # save plot
    out_dir_path = get_attention_head_plots_dir(result.key)
    os.makedirs(out_dir_path, exist_ok=True)
    out_path = os.path.join(out_dir_path, 'js.json')
    with open(out_path, 'w') as f:
        f.write(result.model_dump_json(indent=2))

    # save tensors
    tensor_dir_path = os.path.join(out_dir_path, 'tensors')
    os.makedirs(tensor_dir_path, exist_ok=True)
    mp.save_to_dir(tensor_dir_path)

    # save inputs
    # inputs = []
    # for i, x in enumerate(ds):
    #     if limit is not None and i >= limit:
    #         break
    #     text = x[ds_info.text_column]
    #     tokenized = agent_state.api.tokenize_text(text)
    #     inputs.append({
    #         'text': text,
    #         'token_ids': tokenized.token_ids,
    #         'token_names': tokenized.token_names,
    #     })
    #
    ds.save_to_disk(os.path.join(out_dir_path, 'inputs'))

    return {
        'key': result.key,
    }
