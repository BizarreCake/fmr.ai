from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple
from pydantic import BaseModel
from datasets import Dataset

from fmrai.tracker import TensorId


class AgentTextPrediction(BaseModel):
    token_ids: List[int]
    token_names: List[str]


class AgentDatasetInfo(BaseModel):
    name: str
    text_column: Optional[str] = None
    description: Optional[str] = None


class AgentDatasetList(BaseModel):
    datasets: List[AgentDatasetInfo]


class AgentAPI:
    def predict_zero(self):
        """
        Perform a prediction on a zero tensor.
        This is used to probe the model's structure.
        """
        raise NotImplementedError()

    def predict_text_one(self, text: str) -> AgentTextPrediction:
        raise NotImplementedError()

    def predict_text_bunch(self, texts: List[str]) -> AgentTextPrediction:
        raise NotImplementedError()

    def predict_text_many(
            self,
            dataset: Dataset,
            text_column: str,
            *,
            limit: Optional[int] = None
    ) -> AgentTextPrediction:
        raise NotImplementedError()

    def list_datasets(self) -> AgentDatasetList:
        return AgentDatasetList(datasets=[])

    def load_dataset(self, name: str) -> Optional[Tuple[Dataset, AgentDatasetInfo]]:
        return None
