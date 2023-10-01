import requests
from fastapi import APIRouter, HTTPException

from fmrai.logging import get_computation_graph_dir
from server.adapters.repository import get_local_project_repository
from server.entrypoints.web import models
import os

router = APIRouter(prefix='/api')


@router.get('/model-graph')
def get_model_graph(project_uuid: str, model_name: str):
    repo = get_local_project_repository()
    project = repo.get_project(project_uuid)
    if project is None:
        raise HTTPException(400, 'project not found')

    graph_dir = get_computation_graph_dir(model_name, root_dir=project.data_root_dir)
    graph_path = f'{graph_dir}/graph.dot'
    if not os.path.exists(graph_path):
        raise HTTPException(404, 'graph not found')

    with open(graph_path, 'r') as f:
        return {
            'dot': f.read()
        }


@router.post('/model-graph/generate')
def generate_model_graph(data: models.GenerateModelGraphIn):
    repo = get_local_project_repository()
    project = repo.get_project(data.project_uuid)
    if project is None:
        raise HTTPException(400, 'project not found')

    agent = project.get_agent(data.agent_uuid)
    if agent is None:
        raise HTTPException(400, 'agent project not found')

    r = requests.post(
        f'{agent.connect_url}/model/generate-graph',
        json={
            'root_dir': project.data_root_dir,
            'model_name': agent.model_name,
        }
    )

    if r.status_code != 200:
        raise HTTPException(400, 'agent error')

    return {
        'status': 'ok'
    }
