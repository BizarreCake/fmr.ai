import uuid

from fastapi import APIRouter

from server.adapters.repository import get_local_project_repository
from server.domain.project import AgentInfo
from server.entrypoints.web import models

router = APIRouter(prefix='/api/projects')


@router.post('/new')
def new_project(data: models.NewProjectIn):
    repo = get_local_project_repository()
    project = repo.new_project(data.name)

    return {
        'uuid': project.uuid
    }


@router.get('/list')
def list_projects():
    repo = get_local_project_repository()
    projects = repo.list_projects()

    return {
        'projects': [project.make_info_model() for project in projects]
    }


@router.get('/get')
def get_project(uuid: str):
    repo = get_local_project_repository()
    project = repo.get_project(uuid)

    if project is None:
        return {'project': None}

    return {
        'project': project.make_info_model()
    }


@router.get('/agents/list')
def list_agents(project_uuid: str):
    repo = get_local_project_repository()
    project = repo.get_project(project_uuid)

    return {
        'agents': project.agents,
    }


@router.post('/agents/add')
def add_agent(data: models.AddAgentIn):
    repo = get_local_project_repository()
    project = repo.get_project(data.project_uuid)

    project.agents.append(AgentInfo(
        uuid=uuid.uuid4().hex,
        name=data.agent_name,
        description=None,
        connect_url=data.connect_url,
        model_name=data.model_name,
    ))

    repo.update_project(project)
