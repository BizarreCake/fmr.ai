from typing import Optional, List, Tuple

import torch
import datasets
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from fmrai.agent import AgentAPI
from fmrai.agent.api import TokenizedText, AgentDatasetList
from fmrai.analysis.common import DatasetInfo


class TransformersAgentAPI(AgentAPI):
    def __init__(
            self,
            model_name: str,
            tokenizer_name: Optional[str] = None,
            use_cuda=True,
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

        if use_cuda:
            if torch.cuda.is_available():
                self.model = self.model.cuda()

        self.__dataset_infos = [
            DatasetInfo(
                name='glue/sst2',
                text_column='sentence',
            )
        ]

    def predict_zero(self):
        return self.model(
            input_ids=torch.full((1, 64), 0, dtype=torch.long).to(self.model.device),
            attention_mask=torch.full((1, 64), 1, dtype=torch.long).to(self.model.device),
        ).pooler_output

    def tokenize_text(self, text: str) -> TokenizedText:
        tokenized = self.tokenizer([text], return_tensors='pt')
        tokenized = tokenized.to(self.model.device)
        return TokenizedText(
            token_ids=tokenized['input_ids'].squeeze().tolist(),
            token_names=self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'].squeeze().tolist()),
        )

    def predict_text_bunch(self, texts: List[str]):
        tokenized = self.tokenizer(texts, return_tensors='pt', padding='longest')
        tokenized = tokenized.to(self.model.device)
        self.model(**tokenized)

    def predict_text_many(self, ds: Dataset, text_column: str, *, limit: int):
        chunk = ds[:limit]
        tokenized = self.tokenizer(chunk[text_column], return_tensors='pt', padding='longest')
        tokenized = tokenized.to(self.model.device)
        self.model(**tokenized)

    def predict_text_one(self, text: str):
        tokenized = self.tokenizer([text], return_tensors='pt')
        tokenized = tokenized.to(self.model.device)
        self.model(**tokenized)

        return TokenizedText(
            token_ids=tokenized['input_ids'].squeeze().tolist(),
            token_names=self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'].squeeze().tolist()),
        )

    def list_datasets(self) -> AgentDatasetList:
        return AgentDatasetList(
            datasets=self.__dataset_infos,
        )

    def load_dataset(self, name: str) -> Optional[Tuple[Dataset, DatasetInfo]]:
        if name == 'glue/sst2':
            return (
                datasets.load_dataset('glue', 'sst2', split='validation'),
                self.__dataset_infos[0],
            )

        assert False
