import os
from dataclasses import dataclass

import torch

from fmrai.agent import AgentState
from fmrai.agent.api import AgentTextPrediction
from fmrai.fmrai import get_fmrai
from fmrai.logging import get_log_dir


@dataclass
class TextPredictionResult:
    activation_map_key: str
    result: AgentTextPrediction


def do_init_model(agent_state: AgentState):
    fmr = get_fmrai()

    with fmr.track_computations() as tracker:
        output = agent_state.api.predict_zero()
        graph = tracker.build_graph(output)

    nice_graph = graph.make_nice()
    nice_graph.save(get_log_dir(), 'computation_graph', save_dot=True)


def do_predict_text(agent_state: AgentState, text: str) -> TextPredictionResult:
    fmr = get_fmrai()

    with fmr.track_computations() as tracker:
        with torch.no_grad():
            result = agent_state.api.predict_text(text)

        mp = tracker.build_map()
        print('mp', len(mp.data))

    map_key = os.urandom(8).hex()
    mp.save(map_key)

    return TextPredictionResult(
        activation_map_key=map_key,
        result=result,
    )
