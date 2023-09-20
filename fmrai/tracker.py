import contextlib
import os
import pickle
import threading
import re
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from typing import Union, Dict, Optional, Callable, Any, Iterable

import networkx as nx
import reai.bert.base
import torch
from torch import nn, Tensor
from torch.nn import Parameter

from tqdm import tqdm

from fmrai.instrument import instrumentation_scope, TensorProxy, add_new_tensor_callback, \
    unwrap_proxy, get_current_instrumentation_state, remove_new_tensor_callback
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


class ComputationTracker:
    """
    Tracks activations of a computation graph.
    """

    def __init__(self):
        self._next_ordinal = 0
        self._current_step = 0
        self._root_model = None
        self._tracking = True

    def set_root_model(self, model):
        self._root_model = model

    def __enter__(self):
        add_new_tensor_callback(self._handle_new_tensor)
        self._grad_fn_to_tensor = {}

        self._id_to_tensor = {}
        self._tensor_to_id = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_new_tensor_callback(self._handle_new_tensor)

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

        # print('_hnt', tensor)
        # tensor.retain_grad()

        u = unwrap_proxy(tensor)

        ordinal = self._next_ordinal
        self._next_ordinal += 1

        tensor_id = OrdinalTensorId(ordinal=ordinal)
        self._id_to_tensor[tensor_id] = tensor
        self._tensor_to_id[id(tensor)] = tensor_id

        if getattr(u, 'grad_fn', None) is not None:
            grad_id = id(tensor._saved_grad_fn)
            inside = self._grad_fn_to_tensor.get(grad_id)

            # print('adding', tensor_id, tensor._saved_grad_fn, '->', id(tensor), ':', tensor._saved_grad_fn.next_functions if hasattr(tensor._saved_grad_fn, 'next_functions') else None)
            self._grad_fn_to_tensor[id(tensor._saved_grad_fn)] = tensor

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

    def build_graph(self, output: TensorProxy) -> 'ComputationGraph':
        g = nx.DiGraph()

        class WorkItemType(IntEnum):
            TENSOR = 0
            GRAD_FN = 1

        @dataclass
        class WorkItem:
            type: WorkItemType
            value: Union[Tensor, TensorProxy, torch.autograd.Function]
            callback: Optional[Callable[[Any], None]]
            kwargs: Optional[Dict[str, Any]] = None

        work = [WorkItem(type=WorkItemType.TENSOR, value=output, callback=None)]

        def visit_grad_fn(f: torch.autograd.Function, **_kwargs):
            assert f is not None

            node = GradFnNode(grad_fn=f)

            # print('visit_grad_fn', f, f.next_functions if hasattr(f, 'next_functions') else None)

            g.add_node(node, label=node.label, style='filled', fillcolor='lightgray')

            if hasattr(f, 'next_functions'):
                for prev_f, i in f.next_functions:
                    if prev_f is None:
                        continue

                    prev_tensor = self._grad_fn_to_tensor.get(id(prev_f))
                    # print('  prev_f', prev_f, id(prev_tensor), prev_tensor._saved_grad_fn if prev_tensor is not None else None)
                    if prev_tensor is not None:
                        prev_grad_fn = prev_tensor._saved_grad_fn
                    else:
                        prev_grad_fn = None

                    if prev_tensor is not None and prev_grad_fn is prev_f:
                        # print('    enq ten', id(prev_tensor), self._tensor_to_id.get(id(prev_tensor)))
                        work.append(WorkItem(
                            type=WorkItemType.TENSOR,
                            value=prev_tensor,
                            callback=lambda n: g.add_edge(n, node)
                        ))
                    else:
                        # print('    enq grad_fn', prev_f)
                        work.append(WorkItem(
                            type=WorkItemType.GRAD_FN,
                            value=prev_f,
                            callback=lambda n: g.add_edge(n, node)
                        ))

            if hasattr(f, 'variable'):
                var = f.variable
                if isinstance(var, Parameter):
                    # find the most qualified name for this parameter
                    seen_names = []
                    for mod in get_current_instrumentation_state().seen_modules:
                        for name, param in mod.named_parameters():
                            if param is var:
                                # add name node
                                seen_names.append(name)

                    # find longest name and use that
                    if seen_names:
                        longest_name = max(seen_names, key=len)

                        def callback(n):
                            g.add_edge(n, node)
                            g.nodes[n]['fillcolor'] = 'thistle'

                        work.append(WorkItem(
                            type=WorkItemType.TENSOR,
                            value=var,
                            kwargs={'tensor_id': NamedTensorId(name=longest_name)},
                            callback=callback
                        ))

                elif isinstance(var, Tensor):
                    work.append(WorkItem(
                        type=WorkItemType.TENSOR,
                        value=var,
                        callback=lambda n: g.add_edge(n, node)
                    ))
                else:
                    raise NotImplementedError()

            return node

        def visit_tensor(
                t: Union[TensorProxy, nn.Parameter],
                *,
                tensor_id: Optional[TensorId] = None
        ):
            assert type(t) in (TensorProxy, nn.Parameter)
            assert t is not None

            u = unwrap_proxy(t)
            if tensor_id is None:
                tensor_id = self._tensor_to_id.get(id(t), self._tensor_to_id.get(id(u)))

            if type(t) is TensorProxy:
                grad_fn = t._saved_grad_fn
            else:
                grad_fn = u.grad_fn

            # print('visit_tensor', type(t), id(t), tensor_id, grad_fn)

            node = TensorNode(
                tensor=u,
                tensor_id=tensor_id,
            )

            kwargs = {
                'style': 'filled',
                'fillcolor': 'white'
            }
            if u.requires_grad and grad_fn is None and not isinstance(t, nn.Parameter):
                kwargs['style'] = 'filled'
                kwargs['fillcolor'] = 'wheat'

            g.add_node(node, label=node.label, shape='box', tensor_id=tensor_id, **kwargs)

            def callback(n):
                g.add_edge(n, node)

            if grad_fn is not None:
                work.append(WorkItem(
                    type=WorkItemType.GRAD_FN,
                    value=grad_fn,
                    callback=callback,
                ))

            return node

        # handle all work to completion
        seen = {}
        while work:
            item = work.pop()
            if id(item.value) in seen:
                # just call callback
                item.callback(seen[id(item.value)])
                continue

            if item.type == WorkItemType.TENSOR:
                node = visit_tensor(item.value, **(item.kwargs or {}))
            elif item.type == WorkItemType.GRAD_FN:
                node = visit_grad_fn(item.value, **(item.kwargs or {}))
            else:
                raise NotImplementedError()

            seen[id(item.value)] = node

            if node is not None and item.callback is not None:
                item.callback(node)

        return ComputationGraph(g=g)

    def build_map(self, *, tensors: Optional[Iterable[TensorId]] = None) -> 'ComputationMap':
        if tensors:
            data = {tensor_id: unwrap_proxy(self._id_to_tensor.get(tensor_id)) for tensor_id in tensors}
        else:
            data = {tensor_id: unwrap_proxy(tensor) for tensor_id, tensor in self._id_to_tensor.items()}

        return EagerComputationMap(data=data)


@contextlib.contextmanager
def tracker_scope():
    tracker = ComputationTracker()
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
class TensorStubNode(GraphNode):
    tensor_id: TensorId
    tensor_size: torch.Size

    def __hash__(self):
        return hash(('tensor_stub', self.tensor_id))

    def __eq__(self, other):
        return isinstance(other, TensorStubNode) and self.tensor_id == other.tensor_id

    @property
    def label(self):
        size = re.sub(r'^torch.Size\((.*?)\)$', r'\1', str(self.tensor_size))
        return f'"{self.tensor_id} {size}"'

    def __repr__(self):
        return f'id({self.tensor_id})'


@dataclass
class TensorNode(GraphNode):
    tensor: Tensor
    tensor_id: TensorId

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
class OpNode(GraphNode):
    op: str

    def __hash__(self):
        return hash(('op', self.op))

    def __eq__(self, other):
        return self is other

    @property
    def label(self):
        return self.op

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
    def save(self, key: str, time_step: int = 0):
        raise NotImplementedError()

    def get(self, tensor_id: TensorId) -> Optional[Tensor]:
        raise NotImplementedError()


@dataclass
class EagerComputationMap(ComputationMap):
    data: Dict[TensorId, Tensor]

    def get(self, tensor_id: TensorId) -> Optional[Tensor]:
        return self.data.get(tensor_id)

    def save(self, key: str, time_step: int = 0):
        root_dir = get_computation_map_dir(key)
        for tensor_id, tensor in self.data.items():
            tensor_name = repr(tensor_id)
            assert tensor_name.startswith('@') or tensor_name.startswith('#')

            log_tensor(tensor, tensor_name[1:], time_step=time_step, root_dir=root_dir)


class LazyComputationMap(ComputationMap):
    def __init__(self, root_dir: str, time_step: int):
        self._root_dir = root_dir
        self._time_step = time_step
        self._data = {}

    @staticmethod
    def load_from(path: str, time_step: int = 0) -> 'LazyComputationMap':
        assert os.path.isdir(path)
        return LazyComputationMap(path, time_step)

    def get(self, tensor_id: TensorId) -> Optional[Tensor]:
        existing = self._data.get(tensor_id)
        if existing is not None:
            return existing

        tensor_name = repr(tensor_id)
        assert tensor_name.startswith('@') or tensor_name.startswith('#')

        tensor_path = os.path.join(self._root_dir, tensor_name[1:], f't{self._time_step}.pt')
        if os.path.isfile(tensor_path):
            with open(tensor_path, 'rb') as f:
                tensor = torch.load(f)
                self._data[tensor_id] = tensor
                return tensor

        return None


@dataclass
class NiceComputationGraph:
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


def nice_grad_fn_label(grad_fn: torch.autograd.Function) -> str:
    cls_name = type(grad_fn).__name__
    return re.sub(f'Backward\d+', '', cls_name).lower()


@dataclass
class ComputationGraph:
    g: nx.DiGraph

    def save_dot(self, out_path: str):
        nx.drawing.nx_pydot.write_dot(self.g, out_path)

    def make_nice(self) -> NiceComputationGraph:
        """
        Cleans up the graph for better viewing.
        """
        g = self.g.copy()

        # g.graph['graph'] = {'rankdir': 'LR', 'ranksep': 0}

        for node in list(g.nodes):
            # remove AccumulateGrad nodes
            if isinstance(node, GradFnNode) and node.grad_fn.__class__.__name__ == 'AccumulateGrad':
                # reconnect edges
                for pred in list(g.predecessors(node)):
                    for succ in list(g.successors(node)):
                        g.add_edge(pred, succ)

                g.remove_node(node)
                continue

            # turn all GradFnNode into OpNode
            elif isinstance(node, GradFnNode):
                op_node = OpNode(op=nice_grad_fn_label(node.grad_fn))
                g.add_node(
                    op_node,
                    label=op_node.label,
                    shape='diamond',
                    style='filled',
                    fillcolor='lightgray'
                )

                for pred in list(g.predecessors(node)):
                    g.add_edge(pred, op_node)
                for succ in list(g.successors(node)):
                    g.add_edge(op_node, succ)

                g.remove_node(node)
                continue

            # convert tensor nodes to stub nodes
            elif isinstance(node, TensorNode):
                stub_node = TensorStubNode(tensor_id=node.tensor_id, tensor_size=node.tensor.size())
                g.add_node(
                    stub_node,
                    label=stub_node.label,
                    shape='box',
                    style='filled',
                    fillcolor='white' if isinstance(node.tensor_id, OrdinalTensorId) else 'thistle'
                )

                for pred in list(g.predecessors(node)):
                    g.add_edge(pred, stub_node)
                for succ in list(g.successors(node)):
                    g.add_edge(stub_node, succ)

                g.remove_node(node)
                continue

        return NiceComputationGraph(g=g)


def test_my_bert():
    torch.manual_seed(42)

    with tracker_scope() as tracker:
        config = reai.bert.base.BertConfig(
            vocab_size=8,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            max_len=16,
            attention_type=reai.bert.base.BertAttentionType.STANDARD,
        )

        model = reai.bert.base.BertModelForMaskedLM(config)
        optimizer = torch.optim.Adam(model.parameters())

        for _ in range(3):
            torch.manual_seed(42)
            x = torch.randint(0, 7, (1, 4))
            labels = torch.randint(0, 7, (1, 4,))
            print(x)

            y = model(x, labels=labels)
            print('loss', y.loss)

            mp = tracker.build_map()
            print(mp)

            y.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tracker.step()

        graph = tracker.build_graph(y.loss)

    # export graph to svg
    nx.drawing.nx_pydot.write_dot(graph, 'bert.dot')
    subprocess.check_call(['dot', '-Tsvg', 'bert.dot', '-o', 'bert.svg'])


def test_gradient_tracker():
    torch.manual_seed(42)
    ref_att = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)

    snapshots = []

    with tracker_scope() as tracker:
        torch.manual_seed(42)
        # ref_att = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)
        # att = ref_att
        att = reai.bert.base.BaseMultiHeadAttention(embed_dim=4, num_heads=1)
        att.copy_weights_from_standard_impl(ref_att)

        optimizer = torch.optim.Adam(att.parameters())

        for i in range(1):
            torch.manual_seed(42)
            x = torch.randn((1, 4, 4), requires_grad=True)
            y = att(x, x, x)[0]

            torch.manual_seed(42)
            clf = nn.Linear(4, 1)
            z = clf(y).sum()

            z.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                snap = tracker.build_map()
                print('snap', len(snap.data))
                print(z)
                snapshots.append(snap)

            tracker.step()

    # # export graph to svg
    # nx.drawing.nx_pydot.write_dot(graph, 'test.dot')
    # subprocess.check_call(['dot', '-Tsvg', 'test.dot', '-o', 'test.svg'])

    # dot = torchviz.make_dot(z, show_attrs=True, show_saved=True)
    # with open('test.svg', 'w') as f:
    #     f.write(dot._repr_image_svg_xml())


def test_track_operation():
    torch.manual_seed(42)

    with tracker_scope() as tracker:
        a = torch.randn((4, 4))
        b = torch.randn((4, 4))
        c = a * b
        d = c + a

        mp = tracker.build_map()
        print(mp)


if __name__ == '__main__':
    test_gradient_tracker()
    # test_track_operation()
    # test_my_bert()
