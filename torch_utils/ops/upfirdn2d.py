"""Custom PyTorch ops for efficient resampling of 2D images (HIP-compatible)."""

import os
import torch
import numpy as np

from .. import custom_ops
from .. import misc
from . import conv2d_gradfix





#----------------------------------------------------------------------------

_plugin = None

def _init():
    """Compile and load the HIP or CUDA plugin automatically."""
    global _plugin
    if _plugin is not None:
        return True

    source_dir = os.path.dirname(__file__)

    # Detect ROCm backend
    is_rocm = torch.version.hip is not None

    if is_rocm:
        _plugin = custom_ops.get_plugin(
            module_name="upfirdn2d_plugin",
            sources=["upfirdn2d.cpp", "upfirdn2d.hip"],
            headers=["upfirdn2d.h"],
            source_dir=source_dir,
            extra_cuda_cflags=["--use_fast_math"],  # used by hipcc too
        )
    else:
        _plugin = custom_ops.get_plugin(
            module_name="upfirdn2d_plugin",
            sources=["upfirdn2d.cpp", "upfirdn2d.cu"],
            headers=["upfirdn2d.h"],
            source_dir=source_dir,
            extra_cuda_cflags=["--use_fast_math"],
        )

    return True

#----------------------------------------------------------------------------

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = int(f.shape[-1])
    fh = int(f.shape[0])
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device("cpu"), normalize=True, flip_filter=False, gain=1, separable=None):
    """Prepare filter for upfirdn2d."""
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    if f.ndim == 0:
        f = f[np.newaxis]

    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)

    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl="cuda"):
    """Pad, upsample, filter, and downsample a batch of 2D images."""
    assert isinstance(x, torch.Tensor)
    assert impl in ["ref", "cuda"]

    if impl == "cuda" and x.device.type == "cuda" and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)

    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation using only PyTorch ops."""
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad

    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0):x.shape[2] - max(-pady1, 0), max(-padx0, 0):x.shape[3] - max(-padx1, 0)]

    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x

#----------------------------------------------------------------------------

_upfirdn2d_cuda_cache = {}

def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast GPU (CUDA/HIP) implementation using compiled custom ops."""
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    class Upfirdn2dGPU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f):
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0)
            assert f.ndim in [1, 2]

            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy,
                                       padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0),
                                       upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1),
                                       1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up,
                                      padding=p, flip_filter=(not flip_filter),
                                      gain=gain).apply(dy, f)
            return dx, None

    _upfirdn2d_cuda_cache[key] = Upfirdn2dGPU
    return Upfirdn2dGPU


#----------------------------------------------------------------------------

def upsample2d(x, f=None, up=2, padding=0, flip_filter=False, gain=1, impl="cuda"):
    """Upsample 2D image batch by integer factor."""
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(
        x, f,
        up=up,
        down=1,
        padding=p,
        flip_filter=flip_filter,
        gain=gain * upx * upy,
        impl=impl,
    )

def downsample2d(x, f=None, down=2, padding=0, flip_filter=False, gain=1, impl="cuda"):
    """Downsample 2D image batch by integer factor."""
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(
        x, f,
        up=1,
        down=down,
        padding=p,
        flip_filter=flip_filter,
        gain=gain,
        impl=impl,
    )

def filter2d(x, f=None, padding=0, flip_filter=False, gain=1, impl="cuda"):
    """Apply 2D FIR filter without resampling."""
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - 1) // 2,
        padx1 + fw // 2,
        pady0 + (fh - 1) // 2,
        pady1 + fh // 2,
    ]
    return upfirdn2d(
        x, f,
        up=1,
        down=1,
        padding=p,
        flip_filter=flip_filter,
        gain=gain,
        impl=impl,
    )
#----------------------------------------------------------------------------

