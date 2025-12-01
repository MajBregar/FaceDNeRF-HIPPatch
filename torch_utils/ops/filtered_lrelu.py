# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import os
import numpy as np
import torch
import warnings

from .. import custom_ops
from .. import misc
from . import upfirdn2d
from . import bias_act

#----------------------------------------------------------------------------

_plugin = None

def _init():
    """Load the ROCm/HIP filtered_lrelu plugin instead of CUDA version."""
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='filtered_lrelu_hip_plugin',
            sources=[
                'filtered_lrelu.cpp',
                'filtered_lrelu_wr.hip',
                'filtered_lrelu_rd.hip',
                'filtered_lrelu_ns.hip',
                'filtered_lrelu.hip'
            ],
            headers=['filtered_lrelu.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=[
                '-ffast-math',
                '-D__HIP_PLATFORM_AMD__',
            ],
            extra_ldflags=[],
        )
    return True

#----------------------------------------------------------------------------

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0]  # width, height


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1

#----------------------------------------------------------------------------

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0,
                   gain=np.sqrt(2), slope=0.2, clamp=None,
                   flip_filter=False, impl='hip'):
    """Filtered leaky ReLU for a batch of 2D images (HIP backend)."""
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'hip']
    if impl == 'hip' and x.device.type == 'cuda' and _init():
        # ROCm still reports device type 'cuda'
        return _filtered_lrelu_hip(
            up=up, down=down, padding=padding, gain=gain, slope=slope,
            clamp=clamp, flip_filter=flip_filter).apply(
                x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(
        x, fu=fu, fd=fd, b=b, up=up, down=down,
        padding=padding, gain=gain, slope=slope,
        clamp=clamp, flip_filter=flip_filter)

#----------------------------------------------------------------------------

@misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0,
                        gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    """Reference PyTorch version (unchanged)."""
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    batch_size, channels, in_h, in_w = x.shape
    in_dtype = x.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

    x = bias_act.bias_act(x=x, b=b)
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up,
                            padding=[px0, px1, py0, py1],
                            gain=up**2, flip_filter=flip_filter)
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope,
                          gain=gain, clamp=clamp)
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down,
                            flip_filter=flip_filter)

    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x

#----------------------------------------------------------------------------

_filtered_lrelu_hip_cache = dict()

def _filtered_lrelu_hip(up=1, down=1, padding=0, gain=np.sqrt(2),
                        slope=0.2, clamp=None, flip_filter=False):
    """Fast HIP implementation of filtered_lrelu() using custom ops."""
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_hip_cache:
        return _filtered_lrelu_hip_cache[key]

    class FilteredLReluHIP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)

            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            if si is None:
                si = torch.empty([0])
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

            write_signs = (si.numel() == 0) and (x.requires_grad or b.requires_grad)

            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn("low-performance memory layout detected in filtered_lrelu input", RuntimeWarning)

            if x.dtype in [torch.float16, torch.float32]:
                y, so, return_code = _plugin.filtered_lrelu(
                    x, fu, fd, b, si, up, down, px0, px1, py0, py1,
                    sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1

            if return_code < 0:
                warnings.warn("filtered_lrelu (HIP) fallback path used", RuntimeWarning)
                y = x.add(b.unsqueeze(-1).unsqueeze(-1))
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up,
                                        padding=[px0, px1, py0, py1],
                                        gain=up**2, flip_filter=flip_filter)
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain,
                                                 slope, clamp, write_signs)
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down,
                                        flip_filter=flip_filter)

            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = db = dfu = dfd = dsi = dsx = dsy = None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up ** 2) / (down ** 2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0]  - 1) + py0
                dx = _filtered_lrelu_hip(up=down, down=up, padding=pp,
                                         gain=gg, slope=slope, clamp=None,
                                         flip_filter=ff).apply(
                    dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    _filtered_lrelu_hip_cache[key] = FilteredLReluHIP
    return FilteredLReluHIP
