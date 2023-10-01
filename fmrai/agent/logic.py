import os
from dataclasses import dataclass
from typing import Optional

import datasets
import torch
from pydantic import BaseModel

from fmrai.agent import AgentState
from fmrai.agent.api import TokenizedText
from fmrai.analysis.attention import AttentionHeadClusteringResult
from fmrai.analysis.attention import compute_attention_head_clustering
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.fmrai import get_fmrai
from fmrai.logging import get_attention_head_plots_dir, get_computation_graph_dir, get_computation_map_dir


@dataclass
class TextPredictionResult:
    activation_map_key: str
    result: TokenizedText


def do_generate_model_graph(agent_state: AgentState, *, root_dir: str, model_name: str):
    fmr = get_fmrai()

    with fmr.track() as tracker:
        output = agent_state.api.predict_zero()
        graph = tracker.build_graph(output)

    nice_graph = graph.make_nice()

    out_dir = get_computation_graph_dir(model_name, root_dir=root_dir)
    os.makedirs(out_dir, exist_ok=True)

    nice_graph.save(out_dir, 'graph', save_dot=True)


def do_predict_text(
        agent_state: AgentState,
        text: str,
        *,
        root_dir: str,
) -> TextPredictionResult:
    fmr = get_fmrai()

    with fmr.track() as tracker:
        with torch.no_grad():
            result = agent_state.api.predict_text_one(text)
        mp = tracker.build_map()

    map_key = os.urandom(8).hex()
    out_dir = get_computation_map_dir(map_key, root_dir=root_dir)
    os.makedirs(out_dir, exist_ok=True)

    mp.save_to_dir(out_dir)

    return TextPredictionResult(
        activation_map_key=map_key,
        result=result,
    )


class AttentionHeadPoint(BaseModel):
    x: float
    y: float


def do_compute_attention_head_plot(
        agent_state: AgentState,
        dataset_name: str,
        limit: Optional[int],
        *,
        root_dir: str,
):
    ds, ds_info = agent_state.api.load_dataset(dataset_name)
    if limit is not None:
        ds = ds.select(range(limit))

    fmr = get_fmrai()

    # find attention heads first
    with fmr.track() as tracker:
        y = agent_state.api.predict_zero()
        g = tracker.build_graph(y).make_nice()

    heads = list(find_multi_head_attention(g))
    attention_tensor_ids = [h.softmax_value.tensor_id for h in heads]

    with fmr.track(track_tensors=attention_tensor_ids) as tracker:
        with torch.no_grad():
            agent_state.api.predict_text_many(ds, ds_info.text_column, limit=limit)
        mp = tracker.build_map()

    result = compute_attention_head_clustering(mp, attention_tensor_ids)
    result.dataset_info = ds_info
    result.limit = limit

    # save plot
    out_dir_path = get_attention_head_plots_dir(result.key, root_dir=root_dir)
    os.makedirs(out_dir_path, exist_ok=True)
    out_path = os.path.join(out_dir_path, 'js.json')
    with open(out_path, 'w') as f:
        f.write(result.model_dump_json(indent=2))

    # save tensors
    tensor_dir_path = os.path.join(out_dir_path, 'tensors')
    os.makedirs(tensor_dir_path, exist_ok=True)
    mp.save_to_dir(tensor_dir_path)

    # save inputs
    ds.save_to_disk(os.path.join(out_dir_path, 'inputs'))

    return {
        'key': result.key,
    }


def do_list_attention_head_plot_inputs(
        agent_state: AgentState,
        key: str,
        limit: Optional[int],
        *,
        root_dir: str,
):
    with open(os.path.join(get_attention_head_plots_dir(key, root_dir=root_dir), 'js.json')) as f:
        ds_info = AttentionHeadClusteringResult.model_validate_json(f.read()).dataset_info

    ds_dir = os.path.join(get_attention_head_plots_dir(key, root_dir=root_dir), 'inputs')
    ds = datasets.load_from_disk(ds_dir)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # tokenize
    def tokenizer(x):
        tokenization = agent_state.api.tokenize_text(x[ds_info.text_column])
        return {
            'token_ids': tokenization.token_ids,
            'token_names': tokenization.token_names,
        }

    tok_ds = ds.map(tokenizer)

    return {
        'inputs': [
            {
                'text': x[ds_info.text_column],
                'token_ids': x['token_ids'],
                'token_names': x['token_names'],
            }
            for x in tok_ds
        ]
    }
