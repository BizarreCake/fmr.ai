from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """
    Connection information to an agent.
    """
    name: str
    description: Optional[str]
    connect_url: str


class ProjectInfoModel(BaseModel):
    uuid: str
    name: str
    description: Optional[str]


@dataclass
class Project:
    """
    Project domain object.
    Projects encapsulate the collective analysis of one or more models for some purpose.
    """
    uuid: str
    name: str
    description: Optional[str]
    agents: List[AgentInfo]

    def make_info_model(self) -> ProjectInfoModel:
        return ProjectInfoModel(
            uuid=self.uuid,
            name=self.name,
            description=self.description,
        )

    @staticmethod
    def from_info_model(model: ProjectInfoModel) -> 'Project':
        return Project(
            uuid=model.uuid,
            name=model.name,
            description=model.description,
            agents=[],
        )
