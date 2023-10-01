from typing import List, Optional, Tuple

from datasets import Dataset
from pydantic import BaseModel

from fmrai.analysis.common import DatasetInfo


class TokenizedText(BaseModel):
    token_ids: List[int]
    token_names: List[str]


class AgentDatasetList(BaseModel):
    datasets: List[DatasetInfo]


class AgentAPI:
    def predict_zero(self):
        """
        Perform a prediction on a zero tensor.
        This is used to probe the model's structure.
        """
        raise NotImplementedError()

    def tokenize_text(self, text: str) -> TokenizedText:
        raise NotImplementedError()

    def predict_text_one(self, text: str) -> TokenizedText:
        raise NotImplementedError()

    def predict_text_bunch(self, texts: List[str]) -> TokenizedText:
        raise NotImplementedError()

    def predict_text_many(
            self,
            dataset: Dataset,
            text_column: str,
            *,
            limit: Optional[int] = None
    ) -> TokenizedText:
        raise NotImplementedError()

    def list_datasets(self) -> AgentDatasetList:
        return AgentDatasetList(datasets=[])

    def load_dataset(self, name: str) -> Optional[Tuple[Dataset, DatasetInfo]]:
        return None
