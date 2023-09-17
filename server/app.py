import json
import os.path
import requests

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse

from fmrai.analysis.attention import extract_attention_values
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.logging import get_tensor_info_path, get_log_dir, get_computation_map_dir
from fmrai.tracker import NiceComputationGraph, LazyComputationMap, OrdinalTensorId
from server import models
from server.agent_comm import find_agent_host

app = FastAPI()


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


@app.get('/api/analyze/model/attention')
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
        f'{agent_addr}/predict/text',
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


@app.get('/api/status')
def get_status():
    return {
        'status': 'it works!'
    }
