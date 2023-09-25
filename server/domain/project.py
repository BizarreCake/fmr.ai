from typing import List, Optional


class AgentInfo:
    """
    Connection information to an agent.
    """
    name: str
    description: Optional[str]
    connect_url: str


class Project:
    """
    Project domain object.
    Projects encapsulate the collective analysis of one or more models for some purpose.
    """
    name: str
    description: Optional[str]
    agents: List[AgentInfo]
