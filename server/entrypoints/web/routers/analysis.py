from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Depends, Body

from fmrai.analysis.attention import AttentionHeadClusteringResult, extract_attention_values
from fmrai.analysis.structure import find_multi_head_attention
from fmrai.logging import get_computation_graph_dir, get_attention_head_plots_dir, get_computation_map_dir
from fmrai.tracker import LazyComputationMap, OrdinalTensorId, NiceComputationGraph
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


def _get_project(project_uuid: str):
    repo = get_local_project_repository()
    project = repo.get_project(project_uuid)
    if project is None:
        raise HTTPException(400, 'project not found')

    return project


def _get_agent(project_uuid: str, agent_uuid: str):
    project = _get_project(project_uuid)

    agent = project.get_agent(agent_uuid)
    if agent is None:
        raise HTTPException(400, 'agent project not found')

    return agent


def get_agent_from_params(project_uuid: str, agent_uuid: str):
    return _get_agent(project_uuid, agent_uuid)


def get_agent_from_body(
        project_uuid: str = Body(embed=True),
        agent_uuid: str = Body(embed=True),
):
    return _get_agent(project_uuid, agent_uuid)


def get_project_from_body(project_uuid: str = Body(embed=True)):
    return _get_project(project_uuid)


def get_project_from_params(project_uuid: str):
    return _get_project(project_uuid)


@router.get('/datasets/list')
def list_datasets(agent=Depends(get_agent_from_params)):
    r = requests.get(
        f'{agent.connect_url}/datasets/list',
    )

    return r.json()


@router.get('/analyze/attention/head_plot/list')
def list_attention_head_plots(
        project=Depends(get_project_from_params),
):
    root_dir = get_attention_head_plots_dir(root_dir=project.data_root_dir)

    results = []

    if os.path.isdir(root_dir):
        for key in os.listdir(root_dir):
            with open(os.path.join(root_dir, key, 'js.json')) as f:
                result = AttentionHeadClusteringResult.model_validate_json(f.read())

            results.append({
                'key': key,
                'created_at': result.created_at,
                'dataset_name': result.dataset_info.name,
                'limit': result.limit,
            })

    results.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        'items': results,
    }


@router.post('/analyze/attention/compute_attention_head_plot')
def compute_attention_head_plot(
        project_uuid: str = Body(embed=True),
        agent=Depends(get_agent_from_body),
        dataset_name: str = Body(embed=True),
        limit: Optional[int] = Body(None, embed=True),
):
    repo = get_local_project_repository()
    project = repo.get_project(project_uuid)
    if project is None:
        raise HTTPException(400, 'project not found')

    r = requests.post(
        f'{agent.connect_url}/analyze/attention/head_plot/compute',
        json={
            'dataset': dataset_name,
            'limit': limit,
            'root_dir': project.data_root_dir,
        }
    )

    return r.json()


@router.get('/analyze/attention/head_plot/get')
def get_attention_head_plot(
        key: str,
        project=Depends(get_project_from_params),
):
    plot_dir = get_attention_head_plots_dir(key, root_dir=project.data_root_dir)

    if not os.path.isdir(plot_dir):
        raise HTTPException(status_code=404)

    with open(os.path.join(plot_dir, 'js.json')) as f:
        plot = AttentionHeadClusteringResult.model_validate_json(f.read())

    return {
        'result': plot,
    }


@router.get('/analyze/attention/head_plot/inputs')
def analyze_attention_head_inputs(
        key: str,
        tensor_id: str,
        head_index: int,
        limit: Optional[int] = None,
        project=Depends(get_project_from_params),
        agent=Depends(get_agent_from_params),
):
    plot_dir = get_attention_head_plots_dir(key, root_dir=project.data_root_dir)
    if not os.path.isdir(plot_dir):
        raise HTTPException(status_code=404)

    # get tokenized inputs
    r = requests.post(
        f'{agent.connect_url}/analyze/attention/head_plot/list_inputs',
        json={
            'key': key,
            'limit': limit,
            'root_dir': project.data_root_dir,
        }
    )

    # load tensors
    cmap = LazyComputationMap.load_from(os.path.join(plot_dir, 'tensors'))

    # extract attention
    assert tensor_id.startswith('#')
    tensor_id = OrdinalTensorId(ordinal=int(tensor_id[1:]))
    extraction = extract_attention_values(
        cmap, tensor_id, head_index=head_index, instance_range=range(limit)
    )

    return {
        'inputs': r.json()['inputs'],
        'extraction': extraction,
    }


@router.post('/analyze/text/predict')
def analyze_text_predict(
        project=Depends(get_project_from_body),
        agent=Depends(get_agent_from_body),
        text: str = Body(embed=True)
):
    r = requests.post(
        f'{agent.connect_url}/predict/text/one',
        json={
            'text': text,
            'root_dir': project.data_root_dir,
        }
    )

    r_data = r.json()

    return {
        'key': r_data['activation_map_key'],
        'token_ids': r_data['token_ids'],
        'token_names': r_data['token_names'],
    }


@router.get('/analyze/model/find_attention')
def analyze_model_find_attention(
        model_name: str,
        project=Depends(get_project_from_params),
):
    cg_path = os.path.join(
        get_computation_graph_dir(model_name, root_dir=project.data_root_dir),
        'graph.pickle'
    )
    cg = NiceComputationGraph.load_from(cg_path)
    instances = list(find_multi_head_attention(cg))

    return models.AnalyzeModelFindAttentionOut(
        instances=[
            models.MultiHeadAttentionInstanceModel.from_value(instance)
            for instance in instances
        ]
    )


@router.get('/analyze/text/extract_attention')
def analyze_text_extract_attention(
        key: str,
        tensor_id: str,
        project=Depends(get_project_from_params),
):
    cmap = LazyComputationMap.load_from(get_computation_map_dir(key, root_dir=project.data_root_dir))

    assert tensor_id.startswith('#')
    tensor_id = OrdinalTensorId(ordinal=int(tensor_id[1:]))

    attention_batch = extract_attention_values(cmap, tensor_id)
    return models.AnalyzeTextExtractAttentionOut(
        batch=attention_batch,
    )
