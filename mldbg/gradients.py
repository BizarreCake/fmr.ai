import contextlib
import re
import subprocess
from dataclasses import dataclass
from typing import Union

import networkx as nx
import reai.bert.base
import torch
import reai.bert.base
from torch import nn, Tensor
from torch.nn import Parameter

from mldbg.instrument import instrumentation_scope, TensorProxy, add_new_tensor_callback, \
    unwrap_proxy, get_current_instrumentation_state


@dataclass
class GraphNode:
    pass


@dataclass
class TensorNode(GraphNode):
    tensor: Tensor

    def __hash__(self):
        return hash(('tensor', self.tensor))

    def __eq__(self, other):
        return isinstance(other, TensorNode) and self.tensor is other.tensor

    @property
    def label(self):
        # return 'tensor'
        result = re.sub(r'\s*,\s*grad_fn=[^)]*', '', str(self.tensor))
        result = re.sub(r'\s*,\s*requires_grad=True', '', result)
        result = re.search(r'tensor\((.*)\)', result, flags=re.MULTILINE | re.DOTALL).group(1)
        return result

    def __repr__(self):
        return f'{self.label} @ {id(self.tensor):08x}'


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
class ParamNode(GraphNode):
    name: str
    param: nn.Parameter

    def __hash__(self):
        return hash(('param', self.param))

    def __eq__(self, other):
        return isinstance(other, ParamNode) and self.param is other.param

    @property
    def label(self):
        return self.name

    def __repr__(self):
        return f'{self.label} @ {id(self.param):08x}'


class GradientGraph:
    g: nx.DiGraph


class GradientTracker:
    def __init__(self):
        self._next_ordinal = 0

    def step(self):
        self._next_ordinal = 0

    def register_callbacks(self):
        add_new_tensor_callback(self._handle_new_tensor)

        self._grad_fn_to_tensor = {}
        self._ordinal_to_tensor = {}

    def _handle_new_tensor(self, tensor: TensorProxy):
        print('_hnt', tensor)
        # tensor.retain_grad()

        ordinal = self._next_ordinal
        self._next_ordinal += 1

        self._ordinal_to_tensor[ordinal] = tensor

        if getattr(tensor, 'grad_fn', None) is not None:
            self._grad_fn_to_tensor[tensor.grad_fn] = tensor

    def build_graph(self, output: TensorProxy):
        g = nx.DiGraph()

        def visit_grad_fn(f: torch.autograd.Function):
            node = GradFnNode(grad_fn=f)

            g.add_node(node, label=node.label, style='filled', fillcolor='lightgray')

            if hasattr(f, 'next_functions'):
                for prev_f, i in f.next_functions:
                    prev_tensor = self._grad_fn_to_tensor.get(prev_f)
                    if prev_tensor is not None:
                        prev_node = visit_tensor(prev_tensor)
                    else:
                        prev_node = visit_grad_fn(prev_f)

                    g.add_edge(prev_node, node)

            if hasattr(f, 'variable'):
                var = f.variable
                if isinstance(var, Parameter):
                    prev_node = visit_tensor(var)
                    g.add_edge(prev_node, node)

                    for mod in get_current_instrumentation_state().seen_modules:
                        for name, param in mod.named_parameters():
                            if param is var:
                                # add name node
                                qualified_name = type(mod).__name__ + '.' + name
                                name_node = ParamNode(name=qualified_name, param=param)
                                g.add_node(name_node, label=name_node.label, shape='box', style='filled',
                                           fillcolor='lightblue')
                                g.add_edge(name_node, prev_node)
                                break

                elif isinstance(var, Tensor):
                    prev_node = visit_tensor(var)
                    g.add_edge(prev_node, node)
                else:
                    raise NotImplementedError()

            return node

        def visit_tensor(t: Union[Tensor, TensorProxy, nn.Parameter]):
            u = unwrap_proxy(t)
            node = TensorNode(
                tensor=u,
            )

            kwargs = {}
            if u.requires_grad and u.grad_fn is None and not isinstance(t, nn.Parameter):
                kwargs['style'] = 'filled'
                kwargs['fillcolor'] = 'wheat'

            g.add_node(node, label=node.label, shape='box', **kwargs)

            if t.grad_fn is not None:
                parent_node = visit_grad_fn(t.grad_fn)
                g.add_edge(parent_node, node)

            return node

        visit_tensor(output)
        return g

    def build_map(self):
        return dict(self._ordinal_to_tensor)


@contextlib.contextmanager
def tracker_scope():
    tracker = GradientTracker()
    try:
        with instrumentation_scope():
            tracker.register_callbacks()
            yield tracker
    finally:
        pass


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
    with tracker_scope() as tracker:
        torch.manual_seed(42)

        x = torch.randn((1, 4, 4), requires_grad=True)
        print('x', type(x), x)

        # att = BaseMultiHeadAttention(embed_dim=4, num_heads=1)
        att = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)
        y = att(x, x, x)[0]
        clf = nn.Linear(4, 1)
        z = clf(y).sum()

        # z = SimpleModel()(x)
        # print('z', type(z), z)

        graph = tracker.build_graph(z)

    # export graph to svg
    nx.drawing.nx_pydot.write_dot(graph, 'test.dot')
    subprocess.check_call(['dot', '-Tsvg', 'test.dot', '-o', 'test.svg'])

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
    # test_gradient_tracker()
    # test_track_operation()
    test_my_bert()
