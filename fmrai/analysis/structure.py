from dataclasses import dataclass
from typing import Generator

from fmrai.analysis.common import weak_topological_sort
from fmrai.tracker import NiceComputationGraph, OpNode, TensorStubNode
import networkx as nx


@dataclass
class MultiHeadAttentionInstance:
    softmax_value: TensorStubNode
    num_heads: int


def find_multi_head_attention(cg: NiceComputationGraph) -> Generator[MultiHeadAttentionInstance, None, None]:
    # search for softmax nodes
    softmax = [node for node in cg.g.nodes if isinstance(node, OpNode) and node.op == 'softmax']

    good_softmax = []
    for node in softmax:
        for reverse in (True, False):
            # find nearest bmm ancestor/descendant
            try:
                bmm_ancestor = next(
                    t[1]
                    for t in nx.bfs_edges(cg.g, node, reverse=reverse, depth_limit=10)
                    if isinstance(t[0], OpNode) and t[0].op == 'bmm'
                )
            except StopIteration:
                break

            # there should be only one path from bmm ancestor to softmax node
            src = bmm_ancestor if reverse else node
            dst = node if reverse else bmm_ancestor
            paths = list(nx.all_simple_paths(cg.g, src, dst, cutoff=10))
            if len(paths) != 1:
                break
        else:
            # found a softmax node with a single bmm ancestor and descendant
            good_softmax.append(node)

    good_softmax = weak_topological_sort(cg.g, good_softmax)
    for softmax_node in good_softmax:
        succs = list(cg.g.successors(softmax_node))
        if len(succs) != 1:
            continue

        succ = succs[0]
        if not isinstance(succ, TensorStubNode):
            continue

        yield MultiHeadAttentionInstance(
            softmax_value=succ,
            num_heads=succ.tensor_size[1],
        )


def test_find_mha():
    cg = NiceComputationGraph.load_from('./data/computation_graph.pickle')


if __name__ == '__main__':
    test_find_mha()
