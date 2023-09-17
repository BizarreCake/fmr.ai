import os
import shutil
import sys

from torch import nn
import torch

from fmrai import fmrai
from fmrai.agent import run_agent, AgentAPI
from fmrai.agent.api import AgentTextPrediction
from fmrai.fmrai import Fmrai
from fmrai.logging import get_log_dir, log_model_parameters


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.sum()
        return x


def pasten():
    import reai.bert.base
    from transformers import AutoModel

    shutil.rmtree(get_log_dir(), ignore_errors=True)

    with fmrai() as fmr:
        fmr: Fmrai

        # model = SimpleModel()

        # config = reai.bert.base.BertConfig(
        #     vocab_size=8,
        #     hidden_size=4,
        #     num_heads=1,
        #     num_layers=1,
        #     max_len=16,
        #     attention_type=reai.bert.base.BertAttentionType.STANDARD,
        # )
        # model = reai.bert.base.BertModelForMaskedLM(config)
        model = AutoModel.from_pretrained('bert-base-uncased')

        fmr.add_model(model, log_parameters=True)

        with fmr.track_computations() as tracker:
            # x = torch.randn((1, 512))
            x = torch.randint(0, 8, (1, 16))

            # build initial graph
            y = model(x)
            graph = tracker.build_graph(y.pooler_output).make_nice()
            graph.save_dot(os.path.join(get_log_dir(), 'model.dot'))

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            tracker.step()

            # train
            for _ in range(0):
                y = model(x)

                y.backward()
                optimizer.step()
                optimizer.zero_grad()

                print('y', y)
                # tracker.log_tensors(parameters=True, activations=False)
                tracker.step()

            # mp = tracker.build_map()
            # print(len(mp.data))


def create_simple_model_api():
    model = SimpleModel()

    class MyAgentAPI(AgentAPI):
        def predict_zero(self):
            return model(torch.zeros((1, 512)))

        def predict_text(self, text: str):
            # with torch.no_grad():
            x = torch.randn((1, 512))
            model(x)

            raise NotImplementedError()

    return MyAgentAPI()


def create_bert_api():
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    class BertAgentAPI(AgentAPI):
        def predict_zero(self):
            return model(
                input_ids=torch.full((1, 512), 0, dtype=torch.long),
                attention_mask=torch.full((1, 512), 1, dtype=torch.long),
            ).pooler_output

        def predict_text(self, text: str):
            tokenized = tokenizer(text, return_tensors='pt')
            model(**tokenized)

            return AgentTextPrediction(
                token_ids=tokenized['input_ids'].squeeze().tolist(),
                token_names=tokenizer.convert_ids_to_tokens(tokenized['input_ids'].squeeze().tolist()),
            )

    return BertAgentAPI()


def pasten_agent():
    with fmrai():
        # api = create_simple_model_api()
        api = create_bert_api()

        run_agent(api)


if __name__ == '__main__':
    if sys.argv and sys.argv[1] == '--agent':
        pasten_agent()
    else:
        pasten()
