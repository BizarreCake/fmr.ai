import json
import os.path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse

from fmrai.analysis.attention import extract_attention_values, AttentionHeadClusteringResult
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.logging import get_tensor_info_path, get_log_dir, get_computation_map_dir, get_attention_head_plots_dir
from fmrai.tracker import NiceComputationGraph, LazyComputationMap, OrdinalTensorId
from server.agent_comm import find_agent_host
from server.entrypoints.web import models
from server.entrypoints.web.routers.projects import router as projects_router
from server.entrypoints.web.routers.analysis import router as analysis_router

app = FastAPI()
app.include_router(projects_router)
app.include_router(analysis_router)


def _init_model():
    agent_addr = find_agent_host()
    r = requests.post(
        f'{agent_addr}/model/init',
    )

    print(r)


@app.get('/api/model/info')
def get_model_info():
    with open('./data/model.json', 'r') as f:
        return {
            'data': json.load(f)
        }


@app.get('/api/model/graph')
def get_model_graph():
    _init_model()

    with open('./data/computation_graph.dot', 'r') as f:
        return {
            'dot': f.read()
        }


@app.get('/api/analyze/model/find_attention')
def analyze_model_find_attention():
    cg = NiceComputationGraph.load_from(os.path.join(get_log_dir(), 'computation_graph.pickle'))
    instances = list(find_multi_head_attention(cg))

    return models.AnalyzeModelFindAttentionOut(
        instances=[
            models.MultiHeadAttentionInstanceModel.from_value(instance)
            for instance in instances
        ]
    )


def _load_tensor_info(tensor_id: str, time_step: int):
    assert tensor_id.startswith('@') or tensor_id.startswith('#')
    tensor_id = tensor_id[1:]

    info_path = get_tensor_info_path(tensor_id, time_step)
    if os.path.isfile(info_path):
        with open(info_path) as f:
            return json.load(f)
    else:
        return None


@app.get('/api/tensor/info')
def get_tensor(tensor_id: str, time_step: int):
    info = _load_tensor_info(tensor_id, time_step)
    if info is None:
        return {
            'tensor': None,
        }

    return {
        'tensor': info
    }


@app.get('/api/tensor/image')
def get_tensor_image(tensor_id: str, time_step: int):
    info = _load_tensor_info(tensor_id, time_step)
    if info is None:
        raise HTTPException(status_code=404, detail='Tensor not found')

    assert info['path']
    return FileResponse(info['path'])


@app.post('/api/analyze/text/predict')
def analyze_text_predict(text: str = Body(embed=True)):
    agent_addr = find_agent_host()
    r = requests.post(
        f'{agent_addr}/predict/text/one',
        json={
            'text': text,
        }
    )

    r_data = r.json()

    return {
        'key': r_data['activation_map_key'],
        'token_ids': r_data['token_ids'],
        'token_names': r_data['token_names'],
    }


@app.get('/api/analyze/text/extract_attention')
def analyze_text_extract_attention(
        key: str,
        tensor_id: str,
):
    cmap = LazyComputationMap.load_from(get_computation_map_dir(key))

    assert tensor_id.startswith('#')
    tensor_id = OrdinalTensorId(ordinal=int(tensor_id[1:]))

    attention_batch = extract_attention_values(cmap, tensor_id)
    return models.AnalyzeTextExtractAttentionOut(
        batch=attention_batch,
    )


@app.post('/api/analyze/attention/compute_attention_head_plot')
def compute_attention_head_plot(
        dataset_name: str = Body(embed=True),
        limit: Optional[int] = Body(None, embed=True),
):
    agent_addr = find_agent_host()
    r = requests.post(
        f'{agent_addr}/analyze/attention/compute_attention_head_plot',
        json={
            'dataset': dataset_name,
            'limit': limit,
        }
    )

    return r.json()


@app.get('/api/analyze/attention/head_plot/list')
def list_attention_head_plots():
    root_dir = get_attention_head_plots_dir()

    results = []

    if os.path.isdir(root_dir):
        for key in os.listdir(root_dir):
            with open(os.path.join(root_dir, key, 'js.json')) as f:
                result = AttentionHeadClusteringResult.model_validate_json(f.read())

            results.append({
                'key': key,
                'created_at': result.created_at,
                'dataset_name': result.dataset_name,
                'limit': result.limit,
            })

    results.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        'items': results,
    }


@app.get('/api/analyze/attention/head_plot/get')
def get_attention_head_plot(key: str):
    plot_dir = get_attention_head_plots_dir(key)

    if not os.path.isdir(plot_dir):
        raise HTTPException(status_code=404)

    with open(os.path.join(plot_dir, 'js.json')) as f:
        plot = AttentionHeadClusteringResult.model_validate_json(f.read())

    return {
        'result': plot,
    }


@app.get('/api/analyze/attention/head_plot/inputs')
def analyze_attention_head_inputs(
        key: str,
        tensor_id: str,
        head_index: int,
        limit: Optional[int] = None,
):
    print('aahi', key, tensor_id, head_index, limit)
    plot_dir = get_attention_head_plots_dir(key)

    if not os.path.isdir(plot_dir):
        raise HTTPException(status_code=404)


@app.get('/api/datasets/list')
def list_datasets():
    agent_addr = find_agent_host()
    r = requests.get(
        f'{agent_addr}/datasets/list',
    )

    return r.json()


@app.get('/api/status')
def get_status():
    return {
        'status': 'it works!'
    }
