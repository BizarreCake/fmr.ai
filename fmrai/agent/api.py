from dataclasses import dataclass
from typing import List, Optional, Iterable
from pydantic import BaseModel

from fmrai.tracker import TensorId


class AgentTextPrediction(BaseModel):
    token_ids: List[int]
    token_names: List[str]


class AgentDatasetEntry(BaseModel):
    name: str
    description: Optional[str] = None


class AgentDatasetList(BaseModel):
    datasets: List[AgentDatasetEntry]


class AgentAPI:
    def predict_zero(self):
        """
        Perform a prediction on a zero tensor.
        This is used to probe the model's structure.
        """
        raise NotImplementedError()

    def predict_text_one(self, text: str) -> AgentTextPrediction:
        raise NotImplementedError()

    def predict_text_many(self, texts: List[str]) -> AgentTextPrediction:
        raise NotImplementedError()

    def list_datasets(self) -> AgentDatasetList:
        return AgentDatasetList(datasets=[])

    def load_dataset(self, name: str):
        return None
