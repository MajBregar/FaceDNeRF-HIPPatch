/*
 * HIP port of NVIDIA upfirdn2d.h for ROCm PyTorch.
 */

#pragma once
#include <hip/hip_runtime.h>

//------------------------------------------------------------------------
// HIP kernel parameters.

struct upfirdn2d_kernel_params
{
    const void*     x;              // Input tensor
    const float*    f;              // Filter
    void*           y;              // Output tensor

    int2            up;             // Upsampling factor [x, y]
    int2            down;           // Downsampling factor [x, y]
    int2            pad0;           // Padding [x, y]
    int             flip;           // Whether to flip filter
    float           gain;           // Output scaling factor

    int4            inSize;         // [width, height, channel, batch]
    int4            inStride;
    int2            filterSize;     // [width, height]
    int2            filterStride;
    int4            outSize;        // [width, height, channel, batch]
    int4            outStride;
    int             sizeMinor;      // channels * loopMinor
    int             sizeMajor;      // batch * loopMajor

    int             loopMinor;      // Number of minor loops
    int             loopMajor;      // Number of major loops
    int             loopX;          // Number of X loops
    int             launchMinor;    // Launch minor size
    int             launchMajor;    // Launch major size
};

//------------------------------------------------------------------------
// HIP kernel specialization.

struct upfirdn2d_kernel_spec
{
    void*   kernel;     // Kernel function pointer
    int     tileOutW;   // Tile width
    int     tileOutH;   // Tile height
    int     loopMinor;  // Minor loop count
    int     loopX;      // X loop count
};

//------------------------------------------------------------------------
// HIP kernel selection (templated).

template <class T>
upfirdn2d_kernel_spec choose_upfirdn2d_kernel_hip(const upfirdn2d_kernel_params& p);
