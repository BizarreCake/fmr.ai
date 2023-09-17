from dataclasses import dataclass
from typing import List


@dataclass
class AgentTextPrediction:
    token_ids: List[int]
    token_names: List[str]


class AgentAPI:
    def predict_zero(self):
        """
        Perform a prediction on a zero tensor.
        This is used to probe the model's structure.
        """
        raise NotImplementedError()

    def predict_text(self, text: str) -> AgentTextPrediction:
        raise NotImplementedError()


