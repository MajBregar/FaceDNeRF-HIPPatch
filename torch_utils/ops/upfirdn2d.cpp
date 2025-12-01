/*
 * HIP port of NVIDIA upfirdn2d.cpp for ROCm PyTorch.
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include "upfirdn2d.h"

//------------------------------------------------------------------------

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f,
                               int upx, int upy, int downx, int downy,
                               int padx0, int padx1, int pady0, int pady1,
                               bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on GPU (ROCm reports as CUDA)");
    TORCH_CHECK(f.device() == x.device(), "f must reside on same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.dim() == 4 && f.dim() == 2, "x must be rank 4, f rank 2");
    TORCH_CHECK(upx >= 1 && upy >= 1 && downx >= 1 && downy >= 1, "invalid up/down factors");


    auto dev = device_of(x);
    if (dev->type() == c10::DeviceType::CUDA)
        dev = c10::Device(c10::DeviceType::HIP, dev->index());
    const at::hip::OptionalHIPGuard device_guard(dev);

    int outW = ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
    int outH = ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "invalid output size");

    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW},
                                   x.options(), x.suggest_memory_format());

    // Fill kernel parameter struct.
    upfirdn2d_kernel_params p{};
    p.x            = x.data_ptr();
    p.f            = f.data_ptr<float>();
    p.y            = y.data_ptr();
    p.up           = make_int2(upx, upy);
    p.down         = make_int2(downx, downy);
    p.pad0         = make_int2(padx0, pady0);
    p.flip         = flip ? 1 : 0;
    p.gain         = gain;
    p.inSize       = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.inStride     = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.filterSize   = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize      = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.outStride    = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor    = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
    p.sizeMinor    = (p.inStride.z == 1) ? p.inSize.z : 1;

    // Kernel selection.
    upfirdn2d_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_hip", [&]
    {
        spec = choose_upfirdn2d_kernel_hip<scalar_t>(p);
    });

    // Loop parameters.
    p.loopMajor   = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor   = spec.loopMinor;
    p.loopX       = spec.loopX;
    p.launchMinor = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor = (p.sizeMajor - 1) / p.loopMajor + 1;

    // Grid setup.
    dim3 blockSize, gridSize;
    if (spec.tileOutW < 0) // large
    {
        blockSize = dim3(4, 32, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
            (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
            p.launchMajor);
    }
    else // small
    {
        blockSize = dim3(256, 1, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
            (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
            p.launchMajor);
    }

    // Launch HIP kernel.
    void* args[] = {&p};
    hipStream_t stream = at::hip::getCurrentHIPStream();

    hipError_t err = hipLaunchKernel(
        reinterpret_cast<const void*>(spec.kernel),
        gridSize, blockSize, args, 0, stream
    );

    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP post-launch error: ", hipGetErrorString(err));

    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upfirdn2d", &upfirdn2d, "HIP fused upsample-filter-downsample");
}
