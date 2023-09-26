import os
import uuid
from typing import List, Optional

from server.domain.project import Project, ProjectInfoModel

LOCAL_PROJECTS_ROOT_DIR = 'data/projects'


class ProjectRepository:
    def new_project(self, name: str) -> Project:
        raise NotImplementedError()

    def list_projects(self) -> List[Project]:
        raise NotImplementedError()

    def get_project(self, uuid: str) -> Optional[Project]:
        raise NotImplementedError()


class LocalProjectRepository(ProjectRepository):
    def __init__(self):
        self._root_dir = LOCAL_PROJECTS_ROOT_DIR
        os.makedirs(self._root_dir, exist_ok=True)

    def new_project(self, name: str) -> Project:
        project = Project(
            uuid=uuid.uuid4().hex,
            name=name,
            description=None,
            agents=[]
        )

        project_dir = os.path.join(self._root_dir, project.uuid)
        os.makedirs(project_dir, exist_ok=True)

        with open(os.path.join(project_dir, 'project.json'), 'w') as f:
            f.write(project.make_info_model().model_dump_json(indent=2))

        return project

    def list_projects(self) -> List[Project]:
        projects = []

        for project_uuid in os.listdir(self._root_dir):
            project_dir = os.path.join(self._root_dir, project_uuid)

            if not os.path.isdir(project_dir):
                continue

            with open(os.path.join(project_dir, 'project.json')) as f:
                project_info = ProjectInfoModel.model_validate_json(f.read())

            project = Project.from_info_model(project_info)
            projects.append(project)

        return projects

    def get_project(self, uuid: str) -> Optional[Project]:
        project_dir = os.path.join(self._root_dir, uuid)

        if not os.path.isdir(project_dir):
            return None

        with open(os.path.join(project_dir, 'project.json')) as f:
            project_info = ProjectInfoModel.model_validate_json(f.read())

        project = Project.from_info_model(project_info)
        return project


def get_local_project_repository():
    return LocalProjectRepository()
