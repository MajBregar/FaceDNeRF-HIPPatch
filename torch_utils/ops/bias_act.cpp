/*
 * HIP port of NVIDIA bias_act.cpp for ROCm PyTorch.
 * Works with bias_act.hip kernel.
 */

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include "bias_act.h"

// Kernel selector implemented in bias_act.hip
template <class T> void* choose_bias_act_kernel_hip(const bias_act_kernel_params& p);

//------------------------------------------------------------------------

static bool has_same_layout(const torch::Tensor& x, const torch::Tensor& y)
{
    if (x.dim() != y.dim()) return false;
    for (int64_t i = 0; i < x.dim(); i++)
        if (x.size(i) != y.size(i) ||
            (x.size(i) >= 2 && x.stride(i) != y.stride(i)))
            return false;
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor bias_act(torch::Tensor x, torch::Tensor b, torch::Tensor xref,
                              torch::Tensor yref, torch::Tensor dy,
                              int grad, int dim, int act,
                              float alpha, float gain, float clamp)
{
    TORCH_CHECK(x.is_cuda(), "x must be on GPU (ROCm reports as CUDA)");
    TORCH_CHECK(b.numel() == 0 || (b.dtype() == x.dtype() && b.device() == x.device()), "b mismatch");
    TORCH_CHECK(xref.numel() == 0 || (xref.sizes() == x.sizes() && xref.dtype() == x.dtype()), "xref mismatch");
    TORCH_CHECK(yref.numel() == 0 || (yref.sizes() == x.sizes() && yref.dtype() == x.dtype()), "yref mismatch");
    TORCH_CHECK(dy.numel() == 0 || (dy.sizes() == x.sizes() && dy.dtype() == x.dtype()), "dy mismatch");
    TORCH_CHECK(b.dim() == 1, "b must be 1D");
    TORCH_CHECK(b.numel() == 0 || (dim >= 0 && dim < x.dim()), "dim out of range");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(dim), "b length mismatch");
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be dense");

    auto dev = device_of(x);
    if (dev->type() == c10::DeviceType::CUDA)
        dev = c10::Device(c10::DeviceType::HIP, dev->index());
    const at::hip::OptionalHIPGuard device_guard(dev);

    torch::Tensor y = torch::empty_like(x);
    TORCH_CHECK(has_same_layout(y, x), "y layout mismatch");

    bias_act_kernel_params p{};
    p.x     = x.data_ptr();
    p.b     = (b.numel()) ? b.data_ptr() : nullptr;
    p.xref  = (xref.numel()) ? xref.data_ptr() : nullptr;
    p.yref  = (yref.numel()) ? yref.data_ptr() : nullptr;
    p.dy    = (dy.numel()) ? dy.data_ptr() : nullptr;
    p.y     = y.data_ptr();
    p.grad  = grad;
    p.act   = act;
    p.alpha = alpha;
    p.gain  = gain;
    p.clamp = clamp;
    p.sizeX = static_cast<int>(x.numel());
    p.sizeB = static_cast<int>(b.numel());
    p.stepB = (b.numel()) ? static_cast<int>(x.stride(dim)) : 1;

    // Select HIP kernel
    void* kernel = nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "bias_act_hip", [&]
    {
        kernel = choose_bias_act_kernel_hip<scalar_t>(p);
    });
    TORCH_CHECK(kernel, "no HIP kernel found for specified activation");

    // Launch HIP kernel
    p.loopX = 4;
    int blockSize = 128;
    int gridSize  = (p.sizeX - 1) / (p.loopX * blockSize) + 1;

    hipStream_t stream = at::hip::getCurrentHIPStream();
    void* args[] = { &p };

    hipError_t launch_err = hipLaunchKernel(
        reinterpret_cast<void*>(kernel),
        dim3(gridSize), dim3(blockSize),
        args, 0, stream);

    TORCH_CHECK(launch_err == hipSuccess, "HIP kernel launch failed: ",
                hipGetErrorString(launch_err));

    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bias_act", &bias_act, "HIP fused bias+activation");
}
