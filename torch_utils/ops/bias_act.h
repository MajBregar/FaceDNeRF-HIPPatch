/*
 * Unified header for CUDA/HIP bias_act kernel parameters.
 * Works under both NVIDIA CUDA and AMD ROCm HIP backends.
 */

#pragma once
#include <cstddef>   // for nullptr_t

//------------------------------------------------------------------------
// Kernel parameter struct.

struct bias_act_kernel_params
{
    const void* x;      // [sizeX]
    const void* b;      // [sizeB] or NULL
    const void* xref;   // [sizeX] or NULL
    const void* yref;   // [sizeX] or NULL
    const void* dy;     // [sizeX] or NULL
    void*       y;      // [sizeX]

    int         grad;   // 0=forward, 1=1st grad, 2=2nd grad
    int         act;    // activation selector
    float       alpha;
    float       gain;
    float       clamp;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};

//------------------------------------------------------------------------
// Kernel selection declarations.
// CUDA version:
template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);

// HIP version:
template <class T> void* choose_bias_act_kernel_hip(const bias_act_kernel_params& p);

