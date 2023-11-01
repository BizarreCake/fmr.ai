import contextlib
import functools
import inspect
import types
from dataclasses import dataclass
from functools import wraps
from typing import Union, Any, Dict, Callable, Optional, List, Tuple, FrozenSet

import torch
from torch import Tensor, nn

import bitsandbytes


@dataclass
class InstrumentationState:
    original_functions: Dict[str, Callable]
    new_tensor_callbacks: List[Callable[['TensorProxy'], None]]
    seen_modules: List[nn.Module]
    param_to_name: Dict[nn.Parameter, str]
    track_origin: bool = True
    fmr: Optional['Fmrai'] = None
    disabled: int = 0
    origin_counter: int = 0
    call_depth: int = 0


_CURRENT_INSTRUMENTATION_STATE: Optional[InstrumentationState] = None


def get_current_instrumentation_state():
    if _CURRENT_INSTRUMENTATION_STATE is None:
        raise Exception('Not in instrumentation scope')

    return _CURRENT_INSTRUMENTATION_STATE


def add_new_tensor_callback(fn):
    get_current_instrumentation_state().new_tensor_callbacks.append(fn)


def remove_new_tensor_callback(fn):
    state = get_current_instrumentation_state()
    if fn in state.new_tensor_callbacks:
        state.new_tensor_callbacks.remove(fn)


_SHOULD_NOT_WRAP = [
    '__new__',
    '__init__',
    '__setattr__',
    '__getattr__',
    '__getattribute__',
]


class TensorProxyMeta(type):
    def __new__(cls, name, bases, attrs):
        for attr_name in dir(Tensor):
            if attr_name in _SHOULD_NOT_WRAP:
                continue
            if attr_name in attrs:
                # explicitly defined in the proxy class
                continue

            attr = getattr(Tensor, attr_name)
            if not callable(attr):
                continue

            if inspect.ismethod(attr) and getattr(attr, '__self__', None) is not None:
                continue

            attrs[attr_name] = make_proxy_function(attr, unwrap_args=True)

        return super().__new__(cls, name, bases, attrs)

    # def __instancecheck__(cls, instance):
    #     return isinstance(instance._wrapped, Tensor)
    #
    # def __subclasscheck__(cls, subclass):
    #     return issubclass(Tensor, subclass)


def unwrap_proxy(t: Union['TensorProxy', Any], recursive=True):
    if recursive:
        if isinstance(t, list):
            return [unwrap_proxy(x) for x in t]
        elif isinstance(t, tuple):
            return tuple(unwrap_proxy(x) for x in t)

    return t._wrapped if type(t) is TensorProxy else t


def _wrap_in_proxy(t: Tensor, *, origin: Optional['TensorOrigin'] = None):
    assert type(t) is Tensor or isinstance(t, nn.Parameter)

    state = get_current_instrumentation_state()

    # ordinal = state.get_next_ordinal()
    proxy = TensorProxy(t, origin=origin)

    # call callbacks
    for callback in state.new_tensor_callbacks:
        callback(proxy)

    return proxy


def _try_wrap_in_proxy(value, *, op: Optional[str] = None):
    if type(value) is TensorProxy:
        # print('wrap arg tensor proxy', op)
        pass

    if type(value) is Tensor:
        origin = _make_tensor_origin(op, args=(), kwargs={})
        return _wrap_in_proxy(value, origin=origin)

    if isinstance(value, dict):
        return {k: _try_wrap_in_proxy(v) for k, v in value.items()}

    return value


def _wrap_args_in_proxy(args, kwargs, *, op_base: str):
    args = tuple(_try_wrap_in_proxy(a, op=f'wrap${op_base}(.{i}=X)') for i, a in enumerate(args))
    kwargs = {k: _try_wrap_in_proxy(v, op=f'wrap${op_base}({k}=X)') for k, v in kwargs.items()}
    return args, kwargs


def _make_tensor_origin(
        op: str,
        args,
        kwargs,
        state: Optional[InstrumentationState] = None
) -> Optional['TensorOrigin']:
    if state is None:
        state = get_current_instrumentation_state()

    if not state.track_origin:
        return None

    origin_args = tuple(get_proxy_origin(a) for a in args)
    origin_kwargs = frozenset((k, get_proxy_origin(v)) for k, v in kwargs.items())
    origin = TensorOrigin(
        index=state.origin_counter,
        op=op,
        args=origin_args,
        kwargs=origin_kwargs
    )

    state.origin_counter += 1
    return origin


def make_proxy_function(fn, *, unwrap_args=True):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # print('pf', fn.__name__)
        # print('  args', [type(a) for a in args])
        # print('  kwargs', {k: type(v) for k, v in kwargs.items()})

        state = get_current_instrumentation_state()
        if state.disabled > 0:
            # if disabled, do nothing
            return fn(*args, **kwargs)

        # increase depth only when entering pure torch functions.
        # we do this to properly handle nested torch functions properly.
        # for example, when torch.nn.functional.embedding calls torch.embedding.
        is_pure_pytorch_function = unwrap_args
        depth_delta = int(is_pure_pytorch_function)
        state.call_depth += depth_delta

        try:
            if unwrap_args:
                unwrapped_args = tuple(unwrap_proxy(a) for a in args)
                unwrapped_kwargs = {k: unwrap_proxy(v) for k, v in kwargs.items()}
                result = fn(*unwrapped_args, **unwrapped_kwargs)
            else:
                result = fn(*args, **kwargs)
        finally:
            state.call_depth -= depth_delta

        if type(result) is TensorProxy:
            return result

        if isinstance(result, torch.Tensor) and state.call_depth == 0:
            origin = _make_tensor_origin(fn.__name__, args, kwargs, state=state)
            return _wrap_in_proxy(result, origin=origin)

        return result

    return wrapper


@dataclass(frozen=True)
class TensorOrigin:
    index: Optional[int]
    op: str
    args: Tuple[Optional['TensorOrigin']]
    kwargs: FrozenSet[Tuple[str, Optional['TensorOrigin']]]

    def __hash__(self):
        return hash((self.index, self.op))

    def __eq__(self, other):
        return (
            type(self) is type(other) and
            self.index == other.index and
            self.op == other.op
        )


def get_proxy_origin(t: Union['TensorProxy', Any]) -> Optional[TensorOrigin]:
    if type(t) is TensorProxy:
        return t._origin
    return None


class TensorProxy(metaclass=TensorProxyMeta):
    def __init__(
            self,
            wrapped,
            # ordinal: int,
            origin: Optional[TensorOrigin] = None,
            saved_id: Optional[int] = None,
            saved_grad_fn=None,
    ):
        assert type(wrapped) is not TensorProxy
        self._wrapped = wrapped
        # self._ordinal = ordinal
        self._origin = origin
        self._saved_grad_fn = wrapped.grad_fn if saved_grad_fn is None else saved_grad_fn
        self._saved_id = id(wrapped) if saved_id is None else saved_id

        self.__class__ = Tensor

    def __getattr__(self, item):
        return getattr(self._wrapped, item)

    def __repr__(self):
        return repr(self._wrapped)

    def save_proxy(self):
        """ Returns a copy with a detached tensor. """
        return TensorProxy(
            self._wrapped.detach().cpu(),
            origin=self._origin,
            saved_id=self._saved_id,
            saved_grad_fn=self._saved_grad_fn,
        )


_INSTRUMENTABLE_FUNCTIONS = [
    'torch.tensor',
    'torch.rand',
    'torch.randn',
    'torch.matmul',
    'torch.softmax',
    'torch.bmm',
    'torch.tanh',
    'torch.rsqrt',
    'torch.clone',
    'torch.cat',
    'torch.stack',
    'torch.isfinite',
    'torch.masked_select',
    'torch.zeros',
    'torch.ones',
    'torch.embedding',
    'torch.sort',
    'torch.unique',

    'torch.nn.functional.linear',
    'torch.nn.functional.softmax',
    'torch.nn.functional.multi_head_attention_forward',
    'torch.nn.functional.dropout',
    'torch.nn.functional.layer_norm',
    'torch.nn.functional.relu',
    'torch.nn.functional.gelu',
    'torch.nn.functional.silu',
    'torch.nn.functional.cross_entropy',
    'torch.nn.functional.embedding',

    # bitsandbytes:
    'bnb.matmul',
    'bnb.matmul_4bit',
]


def _resolve_module_and_fn(value):
    value_parts = value.split('.')

    if value_parts[0] == 'torch':
        mod = torch
    elif value_parts[0] == 'bnb':
        mod = bitsandbytes
    else:
        assert False

    for part in value_parts[1:-1]:
        mod = getattr(mod, part)

    fn_name = value_parts[-1]
    fn = getattr(mod, fn_name)

    return mod, fn, fn_name


def instrument_pytorch_module():
    original_init = torch.nn.Module.__init__

    def init_module(*args, **kwargs):
        state = get_current_instrumentation_state()

        self = args[0]
        state.seen_modules.append(self)

        original_init(*args, **kwargs)

    torch.nn.Module.__init__ = init_module

    return {
        'torch.nn.Module.__init__': original_init
    }


def instrument_pytorch_parameter():
    original_new = torch.nn.Parameter.__new__

    def new_parameter(*args, **kwargs):
        result = original_new(*args, **kwargs)

        return result

    torch.nn.Parameter.__new__ = new_parameter

    return {
        'torch.nn.Parameter.__new__': original_new
    }


def instrument_pytorch():
    originals = {}

    for value in _INSTRUMENTABLE_FUNCTIONS:
        mod, fn, fn_name = _resolve_module_and_fn(value)
        originals[value] = fn
        setattr(mod, fn_name, make_proxy_function(fn, unwrap_args=True))

    originals.update(instrument_pytorch_module())
    originals.update(instrument_pytorch_parameter())

    return originals


def deinstrument_pytorch(originals: Dict[str, Callable]):
    for value, fn in originals.items():
        mod, _, fn_name = _resolve_module_and_fn(value)
        setattr(mod, fn_name, fn)


@contextlib.contextmanager
def instrumentation_scope(*, track_origin=True):
    global _CURRENT_INSTRUMENTATION_STATE

    if _CURRENT_INSTRUMENTATION_STATE is not None:
        raise Exception('Cannot nest instrumentation scopes')

    state = InstrumentationState(
        original_functions={},
        new_tensor_callbacks=[],
        seen_modules=[],
        param_to_name={},
        track_origin=track_origin,
    )
    state.original_functions = instrument_pytorch()

    _CURRENT_INSTRUMENTATION_STATE = state
    try:
        yield state
    finally:
        deinstrument_pytorch(state.original_functions)
        _CURRENT_INSTRUMENTATION_STATE = None


@contextlib.contextmanager
def pause_instrumentation():
    state = get_current_instrumentation_state()
    state.disabled += 1
    try:
        yield
    finally:
        state.disabled -= 1


class PostInstrumentationProxyMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance._wrapped, type(instance._wrapped))

    # def __subclasscheck__(cls, subclass):
    #     # TODO


def _copy_function_and_change_closure(f, closure_map):
    new_closure = []
    for cell in f.__closure__:
        if cell.cell_contents in closure_map:
            new_closure.append(types.CellType(closure_map[cell.cell_contents]))
        else:
            new_closure.append(cell)

    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=tuple(new_closure),
    )

    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__

    if inspect.ismethod(f):
        g = g.__get__(f.__self__, f.__self__.__class__)

    return g


class PostInstrumentationProxy(metaclass=PostInstrumentationProxyMeta):
    def __init__(self, wrapped):
        self.__dict__['_wrapped'] = wrapped
        self.__dict__['_wrapped_attrs'] = {}

    def __instrument(self, key, value):
        # print('__instrument', key, type(value))
        if key:
            existing = self._wrapped_attrs.get(key)
            if existing is not None:
                return existing

        if _is_wrappable_object(value) and type(value) is not PostInstrumentationProxy:
            value = _post_instrument_wrappable_object(value, key=key)
            if key:
                self._wrapped_attrs[key] = value
            return value

        if callable(value):
            # rebind & wrap
            if inspect.ismethod(value):
                inner = self.__dict__['_wrapped']
                value = value.__get__(self, inner.__class__)
                # print('  rebound', key)

            # unwrap args only if the callable is a pytorch function
            unwrap_args = value.__module__.startswith('torch.') and not value.__module__.startswith('torch.nn.modules.')

            # check if function has any bound functions in its closure
            # this can be problematic, since those functions can invoke
            # methods bound to the original object and not to the proxy.
            closure_changes = {}
            if hasattr(value, '__closure__') and value.__closure__ is not None:
                for cell in value.__closure__:
                    if callable(cell.cell_contents):
                        cf = cell.cell_contents
                        if hasattr(cf, '__self__') and cf.__self__ is self._wrapped:
                            # closure_changes[cf] = self
                            # print('  closure change', key, cf.__name__)
                            closure_changes[cf] = self.__instrument(None, cf)

            if closure_changes:
                # print('copying and changing function', key)
                value = _copy_function_and_change_closure(value, closure_changes)

            value = make_proxy_function(value, unwrap_args=unwrap_args)

            if key:
                self._wrapped_attrs[key] = value
            return value

        return value

    def __call__(self, *args, **kwargs):
        # print('__call__', type(self._wrapped).__name__)
        if isinstance(self._wrapped, nn.Module):
            args, kwargs = _wrap_args_in_proxy(args, kwargs, op_base=type(self._wrapped).__name__)

        return self.__instrument(
            None,
            type(self._wrapped).__call__(self, *args, **kwargs)
        )

    def __getitem__(self, item):
        # value = type(self._wrapped).__getitem__(self, item)
        value = self._wrapped[item]
        return self.__instrument(f'[{item}]', value)

    def __getattr__(self, item):
        # print('getattr', item, 'of', type(self._wrapped).__name__)
        value = getattr(self._wrapped, item)
        return self.__instrument(item, value)

    def __setattr__(self, key, value):
        if key == 'data':
            raise NotImplementedError
        return setattr(self._wrapped, key, value)

    def __repr__(self):
        return f'<instrumented: {repr(self._wrapped)}>'


def _is_wrappable_object(obj):
    return isinstance(obj, (nn.Module, nn.Parameter, Tensor))


def _post_instrument_wrappable_object(obj, *, key: Optional[str] = None):
    if type(obj) is PostInstrumentationProxy or type(obj) is TensorProxy:
        return obj

    if isinstance(obj, (Tensor, nn.Parameter)):
        origin = _make_tensor_origin(f'parameter={key}', args=(), kwargs={})
        return _wrap_in_proxy(obj, origin=origin)

    # print('  _post_instrument_wrappable_object', type(obj))
    return PostInstrumentationProxy(obj)


def instrument_model(model):
    """
    Applies instrumentation to a model that was created outside instrumentation scope.
    """
    if not _is_wrappable_object(model):
        return model
    return _post_instrument_wrappable_object(model)
