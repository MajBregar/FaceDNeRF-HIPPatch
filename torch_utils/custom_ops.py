# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Modified for ROCm compatibility (HIP dual-mode build).

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid
import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief'  # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------

def _get_mangled_gpu_name():
    """Return normalized GPU name for build cache directory."""
    try:
        name = torch.cuda.get_device_name().lower()
    except Exception:
        name = "cpu"

    # ROCm detection
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        name = name.replace(" ", "-") + "-hip"
    else:
        name = name.replace(" ", "-")

    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA/HIP plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    """Compile and load a PyTorch C++/CUDA/HIP extension module."""
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    try:
        # Ensure compiler is available (Windows only)
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(
                    f'Could not find MSVC installation. Check _find_compiler_bindir() in "{__file__}".'
                )
            os.environ['PATH'] += ';' + compiler_bindir

        # Unset TORCH_CUDA_ARCH_LIST to let backend auto-detect architectures.
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        # Detect ROCm backend and sanitize build flags.
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if is_rocm:
            if "extra_cuda_cflags" in build_kwargs:
                build_kwargs["extra_cuda_cflags"] = [
                    f for f in build_kwargs["extra_cuda_cflags"]
                    if not f.startswith("--use_fast_math")
                ]
            os.environ["USE_ROCM"] = "1"
            print(f"[custom_ops] ROCm backend detected — using hipcc for build.")

        # Compute combined hash for caching.
        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)

        if len(all_source_dirs) == 1:
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            source_digest = hash_md5.hexdigest()

            build_top_dir = torch.utils.cpp_extension._get_build_directory(
                module_name, verbose=verbose_build
            )
            cached_build_dir = os.path.join(
                build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}'
            )

            # Separate cache per ROCm/CUDA
            if is_rocm:
                cached_build_dir += "-rocm"

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir)  # atomic move
                except OSError:
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir):
                        raise

            # Compile and load.
            cached_sources = [
                os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources
            ]
            torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=cached_build_dir,
                verbose=verbose_build,
                sources=cached_sources,
                **build_kwargs,
            )
        else:
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                **build_kwargs,
            )

        # Import compiled module.
        module = importlib.import_module(module_name)

    except Exception as e:
        if verbosity == 'brief':
            print('Failed!')
        raise e

    # Print status and cache.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')

    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
