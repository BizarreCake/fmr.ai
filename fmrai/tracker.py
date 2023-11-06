import contextlib
import os
import pickle
import threading
import re
import subprocess
from dataclasses import dataclass
from enum import IntEnum, Enum, auto
from typing import Union, Dict, Optional, Callable, Any, Iterable, List, Tuple, Iterator

import networkx as nx
# import reai.bert.base
import torch
from torch import nn, Tensor
from torch.nn import Parameter

from tqdm import tqdm

from fmrai.instrument import instrumentation_scope, TensorProxy, add_new_tensor_callback, \
    unwrap_proxy, get_current_instrumentation_state, remove_new_tensor_callback, TensorOrigin
from fmrai.logging import log_model_parameters, log_tensor, get_computation_map_dir


@dataclass(frozen=True)
class TensorId:
    pass


@dataclass(frozen=True)
class OrdinalTensorId(TensorId):
    ordinal: int

    def __repr__(self):
        return f'#{self.ordinal}'


@dataclass(frozen=True)
class NamedTensorId(TensorId):
    name: str

    def __repr__(self):
        return f'@{self.name}'


def _get_raw_op_text(origin: TensorOrigin):
    return origin.op + ' ' + ','.join(str(x) for x in origin.args if isinstance(x, (int, float)))


class ComputationTracker:
    def build_map(self):
        raise NotImplementedError()


class SingleComputationTracker(ComputationTracker):
    """
    Tracks activations of a computation graph.
    """

    def __init__(
            self,
            *,
            track_tensors: Optional[Iterable[TensorId]] = None,
    ):
        self._next_ordinal = 0
        self._current_step = 0
        self._root_model = None
        self._tracking = True
        self._tracked_tensors = list(track_tensors) if track_tensors is not None else None

        self._cg: Optional[nx.DiGraph] = None
        self._dbg_wrote_origin = False

    def set_root_model(self, model):
        self._root_model = model

    def __enter__(self):
        add_new_tensor_callback(self._handle_new_tensor)
        self._grad_fn_to_tensor = {}
        self._id_to_tensor = {}
        self._tensor_to_id = {}

        self._grad_fn_to_tensor_node = {}
        self._id_to_tensor_node = {}
        self._cg = nx.DiGraph()
        self._origin_to_tensor_node = {}
        return self

    @property
    def num_seen_tensors(self):
        return len(self._id_to_tensor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_new_tensor_callback(self._handle_new_tensor)

    # def get_last_tensor(self) -> Optional[Tuple[OrdinalTensorId, TensorProxy]]:
    #     """
    #     Returns the tensors with the highest ordinal id.
    #     """
    #     result = max(
    #         self._id_to_tensor.items(),
    #         key=lambda p: p[0].ordinal if isinstance(p[0], OrdinalTensorId) else 0
    #     )
    #
    #     if not isinstance(result[0], OrdinalTensorId):
    #         return None
    #
    #     return result

    @contextlib.contextmanager
    def no_track(self):
        """
        Starts a new scope in which tensors are not tracked.
        """
        prev_tracking = self._tracking
        self._tracking = False
        try:
            yield
        finally:
            self._tracking = prev_tracking

    def _handle_new_tensor(self, tensor: TensorProxy):
        if not self._tracking:
            return

        ordinal = self._next_ordinal
        self._next_ordinal += 1

        tensor_id = OrdinalTensorId(ordinal=ordinal)
        self._id_to_tensor[tensor_id] = tensor.save_proxy()
        self._tensor_to_id[tensor._saved_id] = tensor_id

        unwraped = unwrap_proxy(tensor)
        tensor_node = TensorNode(tensor=unwraped, tensor_id=tensor_id, tensor_size=unwraped.size())
        self._id_to_tensor_node[tensor_id] = tensor_node
        self._cg.add_node(tensor_node, label=tensor_node.label, shape='box', tensor_id=tensor_id)

        # if ordinal < 10:
        #     print('hnt', ordinal, tensor._origin)

        #
        # use tensor origin to continue graph
        #
        origin: Optional[TensorOrigin] = tensor._origin
        if origin is not None:
            op_node = RawOpNode(op=_get_raw_op_text(origin), origin=origin)
            self._cg.add_node(op_node, label=op_node.label, style='filled', fillcolor='lightgray')
            self._cg.add_edge(op_node, tensor_node)

            self._origin_to_tensor_node[origin] = tensor_node

            prev_origins = set(filter(None, list(origin.args) + list(p[1] for p in origin.kwargs)))
            for prev_origin in prev_origins:
                prev_node = self._origin_to_tensor_node.get(prev_origin)
                if prev_node is not None:
                    self._cg.add_edge(prev_node, op_node)

    def reset(self, *, inc_step=False):
        self._next_ordinal = 0
        self._id_to_tensor.clear()
        self._tensor_to_id.clear()

        if inc_step:
            self._current_step += 1
        else:
            self._current_step = 0

    def step(self):
        self.reset(inc_step=True)

    def get_current_step(self) -> int:
        return self._current_step

    def log_parameters(self):
        if self._root_model is None:
            raise Exception('Cannot log parameters without a set root model (call set_root_model)')

        with self.no_track():
            log_model_parameters(self._root_model, time_step=self.get_current_step())

    def log_activations(self):
        raise NotImplementedError()

    def log_tensors(self, *, parameters=True, activations=True):
        if parameters:
            self.log_parameters()
        if activations:
            self.log_activations()

    def build_graph(
            self,
            y: Optional[TensorProxy] = None,
            *,
            keep_tensors=False,
            limit=None,
    ) -> 'NiceComputationGraph':
        if y is not None:
            raise NotImplementedError('y is not supported yet')

        raw_graph = RawComputationGraph(g=self._cg)

        if limit is not None:
            raw_graph = raw_graph.make_small(limit=limit)

        return NiceComputationGraph.from_raw(raw_graph, keep_tensors=keep_tensors)

    def build_map(self) -> 'ComputationMap':
        if self._tracked_tensors:
            data = {tensor_id: unwrap_proxy(self._id_to_tensor.get(tensor_id)) for tensor_id in self._tracked_tensors}
        else:
            data = {tensor_id: unwrap_proxy(tensor) for tensor_id, tensor in self._id_to_tensor.items()}

        return EagerComputationMap(data=data)


class BatchedComputationTracker(ComputationTracker):
    def __init__(
            self,
            *,
            track_tensors: Optional[Iterable[TensorId]] = None,
    ):
        self._tracker = SingleComputationTracker(track_tensors=track_tensors)
        self._tracked_tensors = list(track_tensors) if track_tensors is not None else None
        self._maps = []

    def __enter__(self):
        self._tracker.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tracker.__exit__(exc_type, exc_val, exc_tb)

    def end_batch(self):
        """ Call this after completing processing a batch. """
        batch_map = self._tracker.build_map()
        self._maps.append(batch_map)

        self._tracker.step()

    def set_root_model(self, model):
        self._tracker.set_root_model(model)

    def build_map(self) -> 'BatchedComputationMap':
        result_maps = self._maps
        self._maps = []

        return BatchedComputationMap(
            batches=result_maps,
        )


@contextlib.contextmanager
def tracker_scope():
    """ DEPRECATED """
    tracker = SingleComputationTracker()
    try:
        with instrumentation_scope():
            tracker._register_callbacks()
            yield tracker
    finally:
        pass


@dataclass
class GraphNode:
    pass


@dataclass
class BaseTensorNode(GraphNode):
    tensor_id: TensorId
    tensor_size: torch.Size

    def __hash__(self):
        return hash(('tensor_base', self.tensor_id))

    def __eq__(self, other):
        return type(other) is type(self) and self.tensor_id == other.tensor_id

    @property
    def label(self):
        size = re.sub(r'^torch.Size\((.*?)\)$', r'\1', str(self.tensor_size))
        return f'"{self.tensor_id} {size}"'

    def __repr__(self):
        return f'id({self.tensor_id})'


@dataclass
class TensorStubNode(BaseTensorNode):
    def __hash__(self):
        return hash(('tensor_stub', self.tensor_id))


@dataclass
class TensorNode(BaseTensorNode):
    tensor: Tensor

    def __hash__(self):
        return hash(('tensor', self.tensor_id, self.tensor))

    def __eq__(self, other):
        return isinstance(other, TensorNode) and self.tensor is other.tensor and self.tensor_id is other.tensor_id

    @property
    def label(self):
        size = re.sub(r'^torch.Size\((.*?)\)$', r'\1', str(self.tensor.size()))
        return f'"{self.tensor_id} {size}"'

        result = re.sub(r'\s*,\s*grad_fn=[^)]*', '', str(self.tensor))
        result = re.sub(r'\s*,\s*requires_grad=True', '', result)
        result = re.search(r'tensor\((.*)\)', result, flags=re.MULTILINE | re.DOTALL).group(1)
        return result

    def __repr__(self):
        return f'ptr({id(self.tensor):08x}) id({self.tensor_id})'


@dataclass
class GradFnNode(GraphNode):
    grad_fn: torch.autograd.Function

    def __hash__(self):
        return hash(('grad_fn', self.grad_fn))

    def __eq__(self, other):
        return isinstance(other, GradFnNode) and self.grad_fn is other.grad_fn

    @property
    def label(self):
        return type(self.grad_fn).__name__

    def __repr__(self):
        return f'{self.label} @ {id(self.grad_fn):08x}'


@dataclass
class RawOpNode(GraphNode):
    op: str
    origin: Optional[TensorOrigin]

    def __hash__(self):
        return hash(('op', self.op))

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return self.op

    def __repr__(self):
        return f'{self.label} @ {id(self):08x}'


class TensorOp(int, Enum):
    SOFTMAX = auto()
    TANH = auto()
    GELU = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()
    ADDMM = auto()
    LINEAR = auto()
    OTHER = auto()


@dataclass
class ConstantNode(GraphNode):
    value: Any

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return str(self.value)

    def __repr__(self):
        return f'{self.label} @ {id(self):08x}'


@dataclass
class OpNode(GraphNode):
    op: TensorOp
    origin: TensorOrigin

    def __hash__(self):
        return hash(('op', self.op))

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return self.op.name.lower()

    def __repr__(self):
        return f'{self.label} @ {id(self):08x}'


# @dataclass
# class ParamNode(GraphNode):
#     name: str
#     param: nn.Parameter
#
#     def __hash__(self):
#         return hash(('param', self.param))
#
#     def __eq__(self, other):
#         return isinstance(other, ParamNode) and self.param is other.param
#
#     @property
#     def label(self):
#         return self.name
#
#     def __repr__(self):
#         return f'{self.label} @ {id(self.param):08x}'


class ComputationMap:
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[TensorId]:
        raise NotImplementedError()

    def save_to_dir(self, dir_path: str, time_step: int = 0):
        raise NotImplementedError()

    def get(self, tensor_id: TensorId) -> List[Tensor]:
        raise NotImplementedError()

    def filter_ids(self, fn: Callable[[TensorId], bool]) -> 'ComputationMap':
        raise NotImplementedError()

    def get_cat(self, tensor_id: TensorId, *, dim=0) -> Optional[Tensor]:
        result = self.get(tensor_id)
        if result is None:
            return None

        return torch.cat(result, dim=dim)


@dataclass
class EagerComputationMap(ComputationMap):
    data: Dict[TensorId, Tensor]

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[TensorId]:
        return iter(self.data.keys())

    def filter_ids(self, fn: Callable[[TensorId], bool]) -> 'ComputationMap':
        return EagerComputationMap(
            data={tensor_id: tensor for tensor_id, tensor in self.data.items() if fn(tensor_id)}
        )

    def get(self, tensor_id: TensorId) -> List[Tensor]:
        tensor = self.data.get(tensor_id)
        if tensor is not None:
            return [tensor]
        return []

    def save_to_dir(self, root_dir: str, time_step: int = 0):
        for tensor_id, tensor in self.data.items():
            tensor_name = repr(tensor_id)
            assert tensor_name.startswith('@') or tensor_name.startswith('#')

            log_tensor(tensor, tensor_name[1:], time_step=time_step, root_dir=root_dir)

    def __repr__(self):
        return f'<eager map: {len(self.data)} tensors>'


class LazyComputationMap(ComputationMap):
    def __init__(self, root_dir: str, time_step: int):
        self._root_dir = root_dir
        self._time_step = time_step
        self._data = {}

    def __len__(self):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def load_from(path: str, time_step: int = 0) -> 'LazyComputationMap':
        assert os.path.isdir(path)
        return LazyComputationMap(path, time_step)

    def get(self, tensor_id: TensorId) -> List[Tensor]:
        existing = self._data.get(tensor_id)
        if existing is not None:
            return [existing]

        tensor_name = repr(tensor_id)
        assert tensor_name.startswith('@') or tensor_name.startswith('#')

        tensor_path = os.path.join(self._root_dir, tensor_name[1:], f't{self._time_step}.pt')
        if os.path.isfile(tensor_path):
            with open(tensor_path, 'rb') as f:
                tensor = torch.load(f)
                self._data[tensor_id] = tensor
                return [tensor]

        return []

    def __repr__(self):
        return f'<lazy map: ? tensors>'


@dataclass
class BatchedComputationMap(ComputationMap):
    batches: List[ComputationMap]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    @staticmethod
    def load_from(path: str, time_step: int = 0) -> 'BatchedComputationMap':
        assert os.path.isdir(path)

        # find all batch directories
        batch_dirs = []
        for name in os.listdir(path):
            if name.startswith('batch_'):
                batch_dir_path = os.path.join(path, name)
                if os.path.isdir(batch_dir_path):
                    batch_dirs.append(batch_dir_path)

        # sort by batch number
        batch_dirs.sort(key=lambda p: int(os.path.basename(p)[len('batch_'):]))

        # load each batch
        batches = []
        for batch_dir_path in tqdm(batch_dirs, desc='Loading batches'):
            batches.append(LazyComputationMap.load_from(batch_dir_path, time_step=time_step))

        return BatchedComputationMap(batches=batches)

    def save_to_dir(self, dir_path: str, time_step: int = 0):
        for i, batch in enumerate(self.batches):
            batch.save_to_dir(
                os.path.join(dir_path, f'batch_{i}'),
                time_step=time_step,
            )

    def save(self, key: str, time_step: int = 0):
        self.save_to_dir(get_computation_map_dir(key), time_step=time_step)

    def get(self, tensor_id: TensorId) -> List[Tensor]:
        result = []
        for batch in self.batches:
            result.extend(batch.get(tensor_id))
        return result


@dataclass
class ComputationGraph:
    g: nx.DiGraph

    def save_dot(self, out_path: str):
        p = nx.drawing.nx_pydot.to_pydot(self.g)
        with open(out_path, 'w') as f:
            f.write(p.to_string())

    def save(self, out_dir_path: str, name: str, *, save_dot=False):
        out_path = os.path.join(out_dir_path, name + '.pickle')
        with open(out_path, 'wb') as f:
            pickle.dump(self, f)

        if save_dot:
            out_path = os.path.join(out_dir_path, name + '.dot')
            self.save_dot(out_path)

    @staticmethod
    def load_from(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


@dataclass
class RawComputationGraph(ComputationGraph):
    g: nx.DiGraph

    def make_small(self, limit: int) -> 'RawComputationGraph':
        to_remove = []

        new_g = self.g.copy()

        # remove all TensorNode instances whose tensor id is an ordinal greater or equal to the limit.
        for node in new_g.nodes:
            if isinstance(node, TensorNode) and isinstance(node.tensor_id, OrdinalTensorId) and node.tensor_id.ordinal >= limit:
                to_remove.append(node)

        # now remove these nodes and all edges connected to them
        new_g.remove_nodes_from(to_remove)

        # remove nodes not connected to anything
        to_remove = []
        for node in new_g.nodes:
            if not list(new_g.predecessors(node)) and not list(new_g.successors(node)):
                to_remove.append(node)

        new_g.remove_nodes_from(to_remove)

        return RawComputationGraph(g=new_g)


def _replace_graph_node(g: nx.DiGraph, old, new, **attrs):
    g.add_node(new, **attrs)

    # add edges from all predecessors to the new node
    for pred in g.predecessors(old):
        g.add_edge(pred, new)

    # add edges from the new node to all successors
    for succ in g.successors(old):
        g.add_edge(new, succ)

    # finally, remove old node
    g.remove_node(old)



_ONE_ARG_OPS = {
    'tanh': TensorOp.TANH,
    'softmax': TensorOp.SOFTMAX,
    'gelu': TensorOp.GELU,
}

_TWO_ARG_OPS = {
    '__add__': TensorOp.ADD,
    '__mul__': TensorOp.MUL,
}

_TWO_ARG_ONE_CONST_OPS = {
    '__radd__': TensorOp.ADD,
    '__rmul__': TensorOp.MUL,
    'pow': TensorOp.POW,
}

_ANY_OPS = {
    'addmm': TensorOp.ADDMM,
    'linear': TensorOp.LINEAR,
}


def _convert_raw_op_node(g: nx.DiGraph, node: RawOpNode):
    op = _ONE_ARG_OPS.get(node.origin.op)
    if op is not None and len(node.origin.args) == 1 and isinstance(node.origin.args[0], TensorOrigin):
        return OpNode(op=op, origin=node.origin)

    op = _TWO_ARG_OPS.get(node.origin.op)
    if op is not None and len(node.origin.args) == 2 and all(isinstance(arg, TensorOrigin) for arg in node.origin.args):
        return OpNode(op=op, origin=node.origin)

    op = _TWO_ARG_ONE_CONST_OPS.get(node.origin.op)
    if op is not None and sum(int(isinstance(arg, (int, float))) for arg in node.origin.args) == 1:
        arg = next(arg for arg in node.origin.args if isinstance(arg, (int, float)))

        # add constant node to graph
        constant_node = ConstantNode(value=arg)
        g.add_node(constant_node, label=constant_node.label, shape='triangle', style='filled', fillcolor='#d9f6ff')

        result_node = OpNode(op=op, origin=node.origin)
        g.add_edge(constant_node, result_node)
        return result_node

    op = _ANY_OPS.get(node.origin.op)
    if op is not None:
        return OpNode(op=op, origin=node.origin)

    return OpNode(op=TensorOp.OTHER, origin=node.origin)


def _get_dot_node_attrs(node):
    label = node.label
    style = '"filled,solid"'
    fillcolor = 'white'

    if isinstance(node, OpNode):
        fillcolor = 'lightgray'
        if node.op == TensorOp.OTHER:
            style = '"filled,dashed"'
            label = _get_raw_op_text(node.origin)

    return dict(
        label=label,
        style=style,
        fillcolor=fillcolor,
    )


class NiceComputationGraph(ComputationGraph):
    @staticmethod
    def from_raw(rg: RawComputationGraph, *, keep_tensors=False):
        g = rg.g.copy()

        for node in list(g.nodes):
            if isinstance(node, RawOpNode):
                op_node = _convert_raw_op_node(g, node)
                _replace_graph_node(g, node, op_node, **_get_dot_node_attrs(op_node))

            elif isinstance(node, TensorNode) and not keep_tensors:
                # replace with stubs
                stub_node = TensorStubNode(tensor_id=node.tensor_id, tensor_size=node.tensor.size())
                _replace_graph_node(g, node, stub_node, **_get_dot_node_attrs(stub_node))

        return NiceComputationGraph(g=g)
