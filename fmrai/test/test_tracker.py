from fmrai import fmrai
from fmrai.agent import AgentAPI
from fmrai.agent.agents.transformers import TransformersAgentAPI
from fmrai.fmrai import Fmrai
from fmrai.logging import get_computation_map_dir
from fmrai.tracker import OrdinalTensorId, BatchedComputationMap


def _create_batch(fmr: Fmrai, agent: AgentAPI):
    with fmr.track(batched=True, track_tensors=[OrdinalTensorId(ordinal=21)]) as tracker:
        agent.predict_text_bunch(['hello world'])
        tracker.end_batch()

        agent.predict_text_bunch(['wassup!'])
        tracker.end_batch()

        return tracker.build_map()


def test_batched_tracking():
    with fmrai() as fmr:
        agent = TransformersAgentAPI('bert-base-uncased')

        result = _create_batch(fmr, agent)

        assert isinstance(result, BatchedComputationMap)
        assert len(result) == 2


def test_load_batch_map():
    with fmrai() as fmr:
        agent = TransformersAgentAPI('bert-base-uncased')

        result = _create_batch(fmr, agent)
        result.save('pasten')

        loaded = BatchedComputationMap.load_from(get_computation_map_dir('pasten'))
        assert isinstance(loaded, BatchedComputationMap)
        assert len(loaded) == 2
        assert len(loaded.get(OrdinalTensorId(ordinal=21))) == 2
