from typing import List

from pydantic import BaseModel

from fmrai.analysis.attention import AttentionExtraction
from fmrai.analysis.structure import MultiHeadAttentionInstance


class NewProjectIn(BaseModel):
    name: str


class AddAgentIn(BaseModel):
    project_uuid: str
    connect_url: str
    agent_name: str
    model_name: str


class GenerateModelGraphIn(BaseModel):
    model_name: str
