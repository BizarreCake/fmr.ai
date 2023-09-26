from fastapi import APIRouter

from server.adapters.repository import get_local_project_repository
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
