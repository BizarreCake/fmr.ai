import os
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """
    Connection information to an agent.
    """
    uuid: str
    name: str
    description: Optional[str]
    connect_url: str
    model_name: str


class ProjectInfoModel(BaseModel):
    uuid: str
    name: str
    description: Optional[str]
    agents: List[AgentInfo]


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
            agents=self.agents,
        )

    @staticmethod
    def load(
            model: ProjectInfoModel,
    ) -> 'Project':
        return Project(
            uuid=model.uuid,
            name=model.name,
            description=model.description,
            agents=model.agents,
        )

    def get_agent(self, agent_uuid: str) -> Optional[AgentInfo]:
        for agent in self.agents:
            if agent.uuid == agent_uuid:
                return agent
        return None

    @property
    def data_root_dir(self):
        return os.path.join('./data/projects', self.uuid)
