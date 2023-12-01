import json
import os.path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from fmrai.logging import get_tensor_info_path
from server.agent_comm import find_agent_host
from server.entrypoints.web.routers.analysis import router as analysis_router
from server.entrypoints.web.routers.projects import router as projects_router
from server.entrypoints.web.routers.kv import router as kv_router

app = FastAPI()
app.include_router(projects_router)
app.include_router(analysis_router)
app.include_router(kv_router)


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


@app.get('/api/status')
def get_status():
    return {
        'status': 'it works!'
    }
