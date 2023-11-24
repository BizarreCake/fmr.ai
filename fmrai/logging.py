import functools
import json
import operator
import os
from typing import Optional

import torch.nn.functional
from torch import Tensor

from fmrai.instrument import unwrap_proxy


def get_log_dir():
    return './data'


def get_computation_graph_dir(model_name: str, *, root_dir: Optional[str] = None):
    p = os.path.join('computation_graphs', model_name)
    if root_dir:
        p = os.path.join(root_dir, p)
    return p


def get_slices_dir():
    return os.path.join(get_log_dir(), 'slices')


def get_slice_dir(time_step: int):
    return os.path.join(get_log_dir(), 'slices', str(time_step))


def get_tensor_dir(name: str):
    return os.path.join(get_log_dir(), 'tensors', name)


def get_tensor_info_path(name: str, time_step: int, *, tensor_dir: Optional[str] = None):
    if tensor_dir is None:
        tensor_dir = get_tensor_dir(name)
    return os.path.join(tensor_dir, f't{time_step}.json')


def get_computation_map_dir(name: str, *, root_dir: Optional[str] = None):
    p = os.path.join('computation_maps', name)
    if root_dir:
        p = os.path.join(root_dir, p)
    return p


def get_attention_head_plots_dir(key: Optional[str] = None, *, root_dir: Optional[str] = None):
    if key is None:
        p = 'attention_head_plots'
    else:
        p = os.path.join(get_attention_head_plots_dir(None), key)

    if root_dir:
        p = os.path.join(root_dir, p)
    return p


def _model_to_json(model):
    parameters = {
        name: {
            'shape': list(param.shape),
        }
        for name, param in model.named_parameters()
    }

    return {
        'parameters': parameters,
    }


def log_model(model):
    os.makedirs(get_log_dir(), exist_ok=True)

    with open(os.path.join(get_log_dir(), 'model.json'), 'w') as f:
        json.dump(_model_to_json(model), f, indent=2)


def log_model_parameters(model, time_step: int):
    for name, param in model.named_parameters():
        log_tensor(param, name, time_step)


MAX_TENSOR_SIZE = 1024


def downscale_tensor(tensor: Tensor):
    s = tensor.size()

    if len(s) == 1:
        # 1d tensor
        w = s[0]
        if w > MAX_TENSOR_SIZE:
            scaling_factor = MAX_TENSOR_SIZE / w
            new_w = max(1, int(w * scaling_factor))

            print('downscaling', w, '->', new_w)

            return torch.nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), (new_w,), mode='nearest').squeeze(0).squeeze(0)

    elif len(s) == 2:
        # 2d tensor
        w = s[0]
        h = s[1]
        longest = max(w, h)
        if longest > MAX_TENSOR_SIZE:
            scaling_factor = MAX_TENSOR_SIZE / longest
            new_w = max(1, int(w * scaling_factor))
            new_h = max(1, int(h * scaling_factor))

            print('downscaling', w, h, '->', new_w, new_h)

            return torch.nn.functional.interpolate(tensor.unsqueeze(0).unsqueeze(0), (new_w, new_h), mode='bilinear').squeeze(0).squeeze(0)

    return tensor


def prepare_image_tensor(tensor: Tensor):
    tensor = downscale_tensor(tensor)

    # scale tensor values to be in range [0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    return tensor.detach().cpu()


def tensor_to_image(tensor: Tensor) -> 'PIL.Image':
    tensor = prepare_image_tensor(tensor)

    if len(tensor.size()) == 1:
        # convert to 2d
        tensor = tensor.unsqueeze(0)

    if len(tensor.size()) != 2:
        return None

    import torchvision.transforms
    transform = torchvision.transforms.ToPILImage()
    img = transform(tensor.unsqueeze(0))
    return img


def log_tensor(
        tensor,
        name: str,
        time_step: int,
        *,
        root_dir: Optional[str] = None,
        formats=None,
):
    if formats is None:
        formats = ['torch']
    used_formats = []

    tensor = unwrap_proxy(tensor)

    if root_dir is None:
        tensor_dir = get_tensor_dir(name)
    else:
        tensor_dir = os.path.join(root_dir, name)

    os.makedirs(tensor_dir, exist_ok=True)

    out_path = get_tensor_info_path(name, time_step, tensor_dir=tensor_dir)
    out_data = {
        'size': list(tensor.size()),
    }

    # save pytorch tensor
    if 'torch' in formats:
        used_formats.append('torch')
        tensor_path = os.path.join(tensor_dir, f't{time_step}.pt')
        out_data['torch'] = tensor_path
        torch.save(tensor, tensor_path)

    if 'image' in formats:
        img = tensor_to_image(tensor)
    else:
        img = None

    if img is not None:
        img_path = os.path.join(tensor_dir, f't{time_step}.jpg')
        img.save(img_path)
        used_formats.append('image')
        out_data.update({
            'image': img_path,
        })

    if 'inline' in formats and functools.reduce(operator.mul, tensor.size(), 1) < 128:
        used_formats.append('inline')
        out_data.update({
            'inline': tensor.tolist(),
        })

    out_data['formats'] = used_formats

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)