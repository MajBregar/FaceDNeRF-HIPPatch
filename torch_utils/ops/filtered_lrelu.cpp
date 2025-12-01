#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include "filtered_lrelu.h"



static std::tuple<torch::Tensor, torch::Tensor, int> filtered_lrelu(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si,
    int up, int down, int px0, int px1, int py0, int py1, int sx, int sy,
    float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
{
    // Ensure HIP device

    auto dev = device_of(x);
    if (dev->type() == c10::DeviceType::CUDA)
        dev = c10::Device(c10::DeviceType::HIP, dev->index());
    const at::hip::OptionalHIPGuard device_guard(dev);

    // Validate arguments
    TORCH_CHECK(fu.device() == x.device() && fd.device() == x.device() && b.device() == x.device(),
                "all input tensors must reside on the same device");
    TORCH_CHECK(fu.dtype() == torch::kFloat && fd.dtype() == torch::kFloat, "fu and fd must be float32");
    TORCH_CHECK(b.dtype() == x.dtype(), "x and b must have the same dtype");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat, "x and b must be float16 or float32");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK((fu.dim() == 1 || fu.dim() == 2) && (fd.dim() == 1 || fd.dim() == 2), "fu and fd must be rank 1 or 2");
    TORCH_CHECK(fu.size(0) <= INT_MAX && fu.size(-1) <= INT_MAX, "fu is too large");
    TORCH_CHECK(fd.size(0) <= INT_MAX && fd.size(-1) <= INT_MAX, "fd is too large");
    TORCH_CHECK(fu.numel() > 0, "fu is empty");
    TORCH_CHECK(fd.numel() > 0, "fd is empty");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(1), "b must be a vector with the same number of channels as x");
    TORCH_CHECK(up >= 1 && down >= 1, "up and down must be at least 1");

    // Query shared memory
    int maxSharedBytes = 0;
    AT_CUDA_CHECK(hipDeviceGetAttribute(&maxSharedBytes, hipDeviceAttributeMaxSharedMemoryPerBlock, x.device().index()));
    int sharedKB = maxSharedBytes >> 10;

    // Kernel params
    filtered_lrelu_kernel_params p{};
    p.up      = up;
    p.down    = down;
    p.fuShape = make_int2((int)fu.size(-1), fu.dim() == 2 ? (int)fu.size(0) : 0);
    p.fdShape = make_int2((int)fd.size(-1), fd.dim() == 2 ? (int)fd.size(0) : 0);


    printf("choose_filtered_lrelu_kernel: up=%d down=%d fu=(%d,%d) fd=(%d,%d) sharedKB=%d\n",
       p.up, p.down, p.fuShape.x, p.fuShape.y, p.fdShape.x, p.fdShape.y, sharedKB);

    filtered_lrelu_kernel_spec test_spec = choose_filtered_lrelu_kernel<float, int32_t, false, false>(p, sharedKB);
    if (!test_spec.exec) {
        printf("No matching HIP kernel found for dtype=%d up=%d down=%d\n", (int)x.scalar_type(), up, down);
        return std::make_tuple(torch::Tensor(), torch::Tensor(), -1);
    }
        

    int64_t sz = (x.dtype() == torch::kHalf) ? 2 : 4;

    // Input geometry
    int64_t xw = (int)x.size(3);
    int64_t xh = (int)x.size(2);
    int64_t fut_w = (int)fu.size(-1) - 1;
    int64_t fut_h = (int)fu.size(0)  - 1;
    int64_t fdt_w = (int)fd.size(-1) - 1;
    int64_t fdt_h = (int)fd.size(0)  - 1;

    int64_t cw = xw * up + (px0 + px1) - fut_w;
    int64_t ch = xh * up + (py0 + py1) - fut_h;
    TORCH_CHECK(cw > fdt_w && ch > fdt_h, "upsampled buffer must be at least the size of downsampling filter");
    TORCH_CHECK(cw <= INT_MAX && ch <= INT_MAX, "upsampled buffer is too large");

    int64_t yw = (cw - fdt_w + (down - 1)) / down;
    int64_t yh = (ch - fdt_h + (down - 1)) / down;
    TORCH_CHECK(yw > 0 && yh > 0, "output must be at least 1x1");
    TORCH_CHECK(yw <= INT_MAX && yh <= INT_MAX, "output is too large");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), yh, yw}, x.options(), x.suggest_memory_format());

    // Sign tensor allocation
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    int64_t sw_active = 0;
    if (writeSigns)
    {
        sw_active = yw * down - (down - 1) + fdt_w;
        int64_t sh = yh * down - (down - 1) + fdt_h;
        int64_t sw = (sw_active + 15) & ~15;
        TORCH_CHECK(sh <= INT_MAX && (sw >> 2) <= INT_MAX, "signs too large");
        s = so = torch::empty({x.size(0), x.size(1), sh, sw >> 2}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }
    else if (readSigns)
        sw_active = s.size(3) << 2;

    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
    }

    // Fill kernel params
    p.x         = x.data_ptr();
    p.y         = y.data_ptr();
    p.b         = b.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.fu        = fu.data_ptr<float>();
    p.fd        = fd.data_ptr<float>();
    p.pad0      = make_int2(px0, py0);
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.flip      = flip_filters ? 1 : 0;
    p.xShape    = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.yShape    = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.sShape    = (readSigns || writeSigns) ? make_int2((int)s.size(3), (int)s.size(2)) : make_int2(0, 0);
    p.sOfs      = make_int2(sx, sy);
    p.swLimit   = (sw_active + 3) >> 2;

    p.xStride   = make_longlong4(sz * x.stride(3), sz * x.stride(2), sz * x.stride(1), sz * x.stride(0));
    p.yStride   = make_longlong4(sz * y.stride(3), sz * y.stride(2), sz * y.stride(1), sz * y.stride(0));
    p.bStride   = sz * b.stride(0);
    p.fuStride  = make_longlong3(fu.stride(-1), fu.dim() == 2 ? fu.stride(0) : 0, 0);
    p.fdStride  = make_longlong3(fd.stride(-1), fd.dim() == 2 ? fd.stride(0) : 0, 0);

    bool index64b = false;
    if (std::abs(p.bStride * x.size(1)) > INT_MAX) index64b = true;

    filtered_lrelu_kernel_spec spec = {0};
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_hip", [&]
    {
        if constexpr (sizeof(scalar_t) <= 4)
        {
            if      (!index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, true,  false>(p, sharedKB);
            else if (!index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, true >(p, sharedKB);
            else if (!index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int32_t, false, false>(p, sharedKB);
            else if ( index64b &&  writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, true,  false>(p, sharedKB);
            else if ( index64b && !writeSigns &&  readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, true >(p, sharedKB);
            else if ( index64b && !writeSigns && !readSigns) spec = choose_filtered_lrelu_kernel<scalar_t, int64_t, false, false>(p, sharedKB);
        }
    });
    TORCH_CHECK(spec.exec, "internal error - HIP kernel not found");

    void* args[] = {&p};
    int bx = spec.numWarps * 32;
    int gx = (p.yShape.x - 1) / spec.tileOut.x + 1;
    int gy = (p.yShape.y - 1) / spec.tileOut.y + 1;
    int gz = p.yShape.z * p.yShape.w;

    if (spec.xrep)
    {
        p.tilesXrep = spec.xrep;
        p.tilesXdim = gx;
        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }
    else
    {
        p.tilesXrep = 0;
        p.tilesXdim = 0;
    }

    // Filter setup kernel
    AT_CUDA_CHECK(hipLaunchKernel(spec.setup, dim3(1), dim3(1024), args, 0, at::hip::getCurrentHIPStream()));

    if      ( writeSigns && !readSigns) AT_CUDA_CHECK((copy_filters<true,  false>(at::hip::getCurrentHIPStream())));
    else if (!writeSigns &&  readSigns) AT_CUDA_CHECK((copy_filters<false, true >(at::hip::getCurrentHIPStream())));
    else if (!writeSigns && !readSigns) AT_CUDA_CHECK((copy_filters<false, false>(at::hip::getCurrentHIPStream())));

    // Shared mem setup
    AT_CUDA_CHECK(hipFuncSetCacheConfig(spec.exec, hipFuncCachePreferShared));
    if (spec.dynamicSharedKB)
        AT_CUDA_CHECK(hipFuncSetAttribute(spec.exec, hipFuncAttributeMaxDynamicSharedMemorySize, spec.dynamicSharedKB << 10));

    const int maxSubGz = 65535;
    for (int zofs = 0; zofs < gz; zofs += maxSubGz)
    {
        p.blockZofs = zofs;
        int subGz = std::min(maxSubGz, gz - zofs);
        AT_CUDA_CHECK(hipLaunchKernel(spec.exec, dim3(gx, gy, subGz), dim3(bx), args, spec.dynamicSharedKB << 10, at::hip::getCurrentHIPStream()));
    }

    return std::make_tuple(y, so, 0);
}



static torch::Tensor filtered_lrelu_act(
    torch::Tensor x, torch::Tensor si, int sx, int sy,
    float gain, float slope, float clamp, bool writeSigns)
{
    auto dev = device_of(x);
    if (dev->type() == c10::DeviceType::CUDA)
        dev = c10::Device(c10::DeviceType::HIP, dev->index());
    const at::hip::OptionalHIPGuard device_guard(dev);

    // Validate arguments.
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX && x.size(2) <= INT_MAX && x.size(3) <= INT_MAX, "x is too large");
    TORCH_CHECK(x.numel() > 0, "x is empty");
    TORCH_CHECK(x.dtype() == torch::kHalf || x.dtype() == torch::kFloat || x.dtype() == torch::kDouble,
                "x must be float16, float32 or float64");

    // Output signs if we don't have sign input.
    torch::Tensor so;
    torch::Tensor s = si;
    bool readSigns = !!s.numel();
    if (writeSigns)
    {
        int64_t sw = x.size(3);
        sw = (sw + 15) & ~15; // Round to multiple of 16.
        s = so = torch::empty({x.size(0), x.size(1), x.size(2), sw >> 2},
                              x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);
    }

    // Validate sign tensor if in use.
    if (readSigns || writeSigns)
    {
        TORCH_CHECK(s.is_contiguous(), "signs must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
        TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
        TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
        TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
        TORCH_CHECK(s.size(2) <= INT_MAX && (s.size(3) << 2) <= INT_MAX, "signs tensor is too large");
    }

    // Initialize HIP kernel parameters.
    filtered_lrelu_act_kernel_params p;
    p.x         = x.data_ptr();
    p.s         = (readSigns || writeSigns) ? s.data_ptr<unsigned char>() : 0;
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.xShape    = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.xStride   = make_longlong4(x.stride(3), x.stride(2), x.stride(1), x.stride(0));
    p.sShape    = (readSigns || writeSigns) ? make_int2((int)s.size(3) << 2, (int)s.size(2)) : make_int2(0, 0);
    p.sOfs      = make_int2(sx, sy);

    // Choose HIP kernel.
    void* func = nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_act_hip", [&]
    {
        if (writeSigns)
            func = choose_filtered_lrelu_act_kernel<scalar_t, true, false>();
        else if (readSigns)
            func = choose_filtered_lrelu_act_kernel<scalar_t, false, true>();
        else
            func = choose_filtered_lrelu_act_kernel<scalar_t, false, false>();
    });
    TORCH_CHECK(func, "internal error - HIP kernel not found");

    // Launch HIP kernel.
    void* args[] = {&p};
    int bx = 128; // 4 warps per block.

    // Logical launch dimensions.
    uint32_t gx = writeSigns ? p.sShape.x : p.xShape.x;
    uint32_t gy = writeSigns ? p.sShape.y : p.xShape.y;
    uint32_t gz = p.xShape.z * p.xShape.w;
    gx = (gx - 1) / bx + 1;

    // Limit grid size.
    const uint32_t gmax = 65535;
    gy = std::min(gy, gmax);
    gz = std::min(gz, gmax);

    // Launch.
    AT_CUDA_CHECK(hipLaunchKernel(func, dim3(gx, gy, gz), dim3(bx), args, 0, at::hip::getCurrentHIPStream()));

    return so;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("filtered_lrelu",      &filtered_lrelu,      "Filtered Leaky ReLU (HIP backend)");
    m.def("filtered_lrelu_act_", &filtered_lrelu_act,  "Filtered Leaky ReLU activation (HIP backend)");
}
