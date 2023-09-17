from dataclasses import dataclass
from typing import Optional

from fmrai.agent import AgentAPI


@dataclass
class AgentState:
    api: Optional[AgentAPI] = None


_GLOBAL_AGENT_STATE: Optional[AgentState] = None


def get_global_agent_state() -> Optional[AgentState]:
    return _GLOBAL_AGENT_STATE


def set_global_agent_state(state: Optional[AgentState]):
    global _GLOBAL_AGENT_STATE
    _GLOBAL_AGENT_STATE = state
