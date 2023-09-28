from typing import List

from pydantic import BaseModel

from fmrai.analysis.attention import AttentionExtraction
from fmrai.analysis.structure import MultiHeadAttentionInstance


class MultiHeadAttentionInstanceModel(BaseModel):
    softmax_value: str
    num_heads: int

    @staticmethod
    def from_value(value: MultiHeadAttentionInstance):
        return MultiHeadAttentionInstanceModel(
            softmax_value=str(value.softmax_value.tensor_id),
            num_heads=value.num_heads,
        )


class AnalyzeModelFindAttentionOut(BaseModel):
    instances: List[MultiHeadAttentionInstanceModel]


class AnalyzeTextExtractAttentionOut(BaseModel):
    batch: List[AttentionExtraction]


class NewProjectIn(BaseModel):
    name: str


class AddAgentIn(BaseModel):
    project_uuid: str
    connect_url: str
    agent_name: str
    model_name: str


class GenerateModelGraphIn(BaseModel):
    project_uuid: str
    agent_uuid: str
    model_name: str
