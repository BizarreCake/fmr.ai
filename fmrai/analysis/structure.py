from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, Iterable, Tuple, Optional, Any, Union
import networkx as nx
from tqdm import tqdm

from fmrai.analysis.common import weak_topological_sort
from fmrai.tracker import NiceComputationGraph, RawOpNode, TensorStubNode, TensorOp, OpNode, ConstantNode, GraphNode, \
    TensorNode, BaseTensorNode


def _find_simple_chain_pattern(cg: NiceComputationGraph, pattern) -> Iterable[Tuple[GraphNode, GraphNode]]:
    def walk(start, node, p):
        # empty pattern, empty match
        if not p:
            return start, node

        # match against constant
        if isinstance(p, (int, float)):
            if isinstance(node, ConstantNode) and abs(node.value - p) < 1e-6:
                return start, node
            return None

        assert isinstance(p, list)
        assert isinstance(p[0], tuple)
        top = p[0]
        op = top[0]
        p_preds = top[1:]

        if isinstance(node, OpNode) and node.op == op:
            node_preds = set(cg.g.predecessors(node))
            used_preds = set()

            for p_pred in p_preds:
                for pred in node_preds - used_preds:
                    if walk(start, pred, p_pred) is not None:
                        # found a match
                        used_preds.add(pred)
                        break
                else:
                    # no match for predecessor
                    return None

            # passed predecessor checks

            # skip tensor node below op node
            succs = list(cg.g.successors(node))
            if len(succs) != 1 or not isinstance(succs[0], (TensorNode, TensorStubNode)):
                return None
            node = succs[0]

            # continue down successor line
            for succ in list(cg.g.successors(node)):
                m = walk(start, succ, p[1:])
                if m is not None:
                    return m

        return None

    for node in cg.g.nodes:
        match = walk(node, node, pattern)
        if match is not None:
            yield match


@dataclass
class MultiHeadAttentionInstance:
    softmax_value: TensorStubNode
    num_heads: int


def find_multi_head_attention(cg: NiceComputationGraph) -> Generator[MultiHeadAttentionInstance, None, None]:
    results = _find_simple_chain_pattern(cg, [
        (TensorOp.SOFTMAX,),
    ])

    softmax = [start for start, _ in results]
    softmax = weak_topological_sort(cg.g, softmax)

    for softmax_node in softmax:
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


_NEW_GELU_PATTERN = [
    (TensorOp.MUL, 0.79788456),
    (TensorOp.TANH,),
    (TensorOp.ADD, 1.0),
    (TensorOp.MUL,),
]


def find_new_gelu(cg: NiceComputationGraph):
    return _find_simple_chain_pattern(cg, _NEW_GELU_PATTERN)


@dataclass
class SearchExtent:
    start: Optional[Any]
    end: Optional[Any]


class GraphPatternFinder:
    def __init__(self, g: nx.DiGraph):
        self.g = g

    def search(self, extent: Optional[SearchExtent] = None) -> Iterable:
        raise NotImplementedError()


class BasicGraphPatternFinder(GraphPatternFinder):
    def check_node(self, node):
        raise NotImplementedError()

    def _iter_searchable_nodes(self, rng: Optional[SearchExtent]):
        # TODO: this is an expensive function that should be cached
        #       for nested finders, this will be called a lot in a redundant manner.
        nodes = set(self.g.nodes)

        if rng is not None:
            if rng.start is not None:
                # include only nodes reachable from start
                nodes = nodes & ({rng.start} | set(nx.descendants(self.g, rng.start)))

            if rng.end is not None:
                # include only nodes that reach end
                nodes = nodes & ({rng.end} | set(nx.ancestors(self.g, rng.end)))

        yield from weak_topological_sort(self.g, nodes)

    def search(self, rng: Optional[SearchExtent] = None):
        for node in self._iter_searchable_nodes(rng):
            m = self.check_node(node)
            if m is not None:
                yield m


class BasicChainFinder(BasicGraphPatternFinder):
    def __init__(self, g: nx.DiGraph, pattern):
        super().__init__(g)
        self.pattern = pattern

    def _walk(self, start, node, p):
        # empty pattern, empty match
        if not p:
            return start, node

        # match against constant
        if isinstance(p, (int, float)):
            if isinstance(node, ConstantNode) and abs(node.value - p) < 1e-6:
                return start, node
            return None

        assert isinstance(p, list)
        assert isinstance(p[0], tuple)
        top = p[0]
        op = top[0]
        p_preds = top[1:]

        if isinstance(node, OpNode) and node.op == op:
            node_preds = set(self.g.predecessors(node))
            used_preds = set()

            for p_pred in p_preds:
                for pred in node_preds - used_preds:
                    if self._walk(start, pred, p_pred) is not None:
                        # found a match
                        used_preds.add(pred)
                        break
                else:
                    # no match for predecessor
                    return None

            # passed predecessor checks

            # skip tensor node below op node
            succs = list(self.g.successors(node))
            if len(succs) != 1 or not isinstance(succs[0], (TensorNode, TensorStubNode)):
                return None
            node = succs[0]

            # continue down successor line
            for succ in list(self.g.successors(node)):
                m = self._walk(start, succ, p[1:])
                if m is not None:
                    return m

        return None

    def check_node(self, node: GraphNode):
        return self._walk(node, node, self.pattern)


class ActivationKind(int, Enum):
    GELU = auto()
    NEW_GELU = auto()


@dataclass
class ActivationInstance:
    kind: ActivationKind
    result_node: Union[BaseTensorNode]


class FindGELU(BasicGraphPatternFinder):
    def check_node(self, node):
        if isinstance(node, OpNode) and node.op == TensorOp.GELU:
            succs = list(self.g.successors(node))
            if len(succs) == 1 and isinstance(succs[0], BaseTensorNode):
                return ActivationInstance(
                    kind=ActivationKind.GELU,
                    result_node=succs[0],
                )


class FindNewGELU(BasicChainFinder):
    def __init__(self, g: nx.DiGraph):
        super().__init__(g, _NEW_GELU_PATTERN)

    def search(self, rng: Optional[SearchExtent] = None):
        matches = list(super().search(rng))

        for _, end_node in matches:
            # go up one node to get the resulting tensor
            preds = list(self.g.predecessors(end_node))
            assert len(preds) == 1
            result_node = preds[0]

            yield ActivationInstance(
                kind=ActivationKind.NEW_GELU,
                result_node=result_node,
            )


class FindActivation(GraphPatternFinder):
    def __init__(self, g):
        super().__init__(g)
        self._finders = [
            FindGELU(g),
            FindNewGELU(g),
        ]

    def search(self, extent: Optional[SearchExtent] = None) -> Generator[ActivationInstance, None, None]:
        for finder in self._finders:
            yield from finder.search(extent)


class FindLinear(BasicGraphPatternFinder):
    def check_node(self, node):
        if isinstance(node, OpNode):
            if node.op in (TensorOp.ADDMM, TensorOp.LINEAR):
                succs = list(self.g.successors(node))
                if len(succs) == 1 and isinstance(succs[0], BaseTensorNode):
                    return succs[0]


def pair_closest_descendants(
        g: nx.DiGraph,
        src_set,
        dst_set,
        *,
        max_dist_error=0,
):
    closest = {}
    for src in tqdm(src_set):
        candidates = [
            (dst, nx.shortest_path_length(g, src, dst))
            for dst in dst_set
            if nx.has_path(g, src, dst)
        ]

        candidates.sort(key=lambda p: p[1])
        if candidates:
            closest[src] = candidates[0]

    if not closest:
        return {}

    min_dist = min(d for _, d in closest.values())
    result_pairs = {
        src: p[0]
        for src, p in closest.items()
        if p[1] <= min_dist + max_dist_error
    }

    return result_pairs


@dataclass
class TransformerFFNInstance:
    linear_top: BaseTensorNode
    act: BaseTensorNode
    linear_bottom: BaseTensorNode


class FindTransformerFFN(GraphPatternFinder):
    def __init__(self, g: nx.DiGraph):
        super().__init__(g)
        self._find_linear = FindLinear(g)
        self._find_act = FindActivation(g)

    def search(self, extent: Optional[SearchExtent] = None):
        linears = list(self._find_linear.search(extent))
        activations = list(self._find_act.search(extent))

        print(len(linears), 'linears')
        print(len(activations), 'activations')

        linear_top_and_act = pair_closest_descendants(
            self.g,
            linears,
            [act.result_node for act in activations],
        )
        act_and_linear_bottom = pair_closest_descendants(
            self.g,
            [act.result_node for act in activations],
            set(linears) - set(linear_top_and_act.keys()),
        )

        print('linear_top_and_act', len(linear_top_and_act))
        print('act_and_linear_bottom', len(act_and_linear_bottom))

        linear_tops = weak_topological_sort(self.g, linear_top_and_act.keys())
        for linear_top in linear_tops:
            act_node = linear_top_and_act.get(linear_top)
            linear_bottom = act_and_linear_bottom.get(act_node)

            yield TransformerFFNInstance(
                linear_top=linear_top,
                act=act_node,
                linear_bottom=linear_bottom,
            )

