from typing import Optional, List

import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from fmrai.agent import AgentAPI
from fmrai.agent.api import TokenizedText


class TransformersAgentAPI(AgentAPI):
    def __init__(
            self,
            model_name: str,
            tokenizer_name: Optional[str] = None,
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    def predict_zero(self):
        return self.model(
            input_ids=torch.full((1, 64), 0, dtype=torch.long),
            attention_mask=torch.full((1, 64), 1, dtype=torch.long),
        ).pooler_output

    def tokenize_text(self, text: str) -> TokenizedText:
        tokenized = self.tokenizer([text], return_tensors='pt')
        return TokenizedText(
            token_ids=tokenized['input_ids'].squeeze().tolist(),
            token_names=self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'].squeeze().tolist()),
        )

    def predict_text_bunch(self, texts: List[str]):
        tokenized = self.tokenizer(texts, return_tensors='pt', padding='longest')
        self.model(**tokenized)

    def predict_text_many(self, ds: Dataset, text_column: str, *, limit: int):
        chunk = ds[:limit]
        tokenized = self.tokenizer(chunk[text_column], return_tensors='pt', padding='longest')
        self.model(**tokenized)

    def predict_text_one(self, text: str):
        tokenized = self.tokenizer([text], return_tensors='pt')
        self.model(**tokenized)

        return TokenizedText(
            token_ids=tokenized['input_ids'].squeeze().tolist(),
            token_names=self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'].squeeze().tolist()),
        )
