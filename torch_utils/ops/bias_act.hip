/*
 * Simplified HIP port of NVIDIA bias_act.cu
 * Copyright NVIDIA; ported for experimental ROCm use.
 */

#include <hip/hip_runtime.h>
#include <c10/util/Half.h>
#include "bias_act.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

//------------------------------------------------------------------------
// HIP kernel.

template <class T, int A>
__global__ void bias_act_kernel_hip(bias_act_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    int G                 = p.grad;
    scalar_t alpha        = (scalar_t)p.alpha;
    scalar_t gain         = (scalar_t)p.gain;
    scalar_t clamp        = (scalar_t)p.clamp;
    scalar_t one          = (scalar_t)1;
    scalar_t two          = (scalar_t)2;
    scalar_t expRange     = (scalar_t)80;
    scalar_t halfExpRange = (scalar_t)40;
    scalar_t seluScale    = (scalar_t)1.0507009873554804934193349852946;
    scalar_t seluAlpha    = (scalar_t)1.6732632423543772848170429916717;

    int xi = blockIdx.x * p.loopX * blockDim.x + threadIdx.x;
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX; loopIdx++, xi += blockDim.x)
    {
        scalar_t x = (scalar_t)((const T*)p.x)[xi];
        scalar_t b = (p.b) ? (scalar_t)((const T*)p.b)[(xi / p.stepB) % p.sizeB] : 0;
        scalar_t xref = (p.xref) ? (scalar_t)((const T*)p.xref)[xi] : 0;
        scalar_t yref = (p.yref) ? (scalar_t)((const T*)p.yref)[xi] : 0;
        scalar_t dy = (p.dy) ? (scalar_t)((const T*)p.dy)[xi] : one;
        scalar_t yy = (gain != 0) ? yref / gain : 0;
        scalar_t y = 0;

        ((G == 0) ? x : xref) += b;

        // linear
        if (A == 1) { y = x; }

        // relu
        if (A == 2) { y = (G == 0 ? (x > 0 ? x : 0) : (yy > 0 ? x : 0)); }

        // lrelu
        if (A == 3) { y = (G == 0 ? (x > 0 ? x : x * alpha) : (yy > 0 ? x : x * alpha)); }

        // tanh
        if (A == 4) {
            if (G == 0) {
                scalar_t c = exp(x), d = one / c;
                y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d);
            } else if (G == 1) y = x * (one - yy * yy);
              else y = x * (one - yy * yy) * (-two * yy);
        }

        // sigmoid
        if (A == 5) {
            if (G == 0) y = (x < -expRange) ? 0 : one / (exp(-x) + one);
            else if (G == 1) y = x * yy * (one - yy);
            else y = x * yy * (one - yy) * (one - two * yy);
        }

        // elu
        if (A == 6) {
            if (G == 0) y = (x >= 0) ? x : exp(x) - one;
            else if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
            else y = (yy >= 0) ? 0 : x * (yy + one);
        }

        // selu
        if (A == 7) {
            if (G == 0) y = (x >= 0) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
            else if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
            else y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
        }

        // softplus
        if (A == 8) {
            if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
            else if (G == 1) y = x * (one - exp(-yy));
            else { scalar_t c = exp(-yy); y = x * c * (one - c); }
        }

        // swish
        if (A == 9) {
            if (G == 0) y = (x < -expRange) ? 0 : x / (exp(-x) + one);
            else {
                scalar_t c = exp(xref), d = c + one;
                if (G == 1) y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
                else y = (xref > halfExpRange) ? 0 : x * c * (xref * (two - d) + two * d) / (d * d * d);
                yref = (xref < -expRange) ? 0 : xref / (exp(-xref) + one) * gain;
            }
        }

        // gain
        y *= gain * dy;

        // clamp
        if (clamp >= 0) {
            if (G == 0)
                y = (y > -clamp && y < clamp) ? y : (y >= 0 ? clamp : -clamp);
            else
                y = (yref > -clamp && yref < clamp) ? y : 0;
        }

        ((T*)p.y)[xi] = (T)y;
    }
}

//------------------------------------------------------------------------
// Kernel selector.

template <class T> void* choose_bias_act_kernel_hip(const bias_act_kernel_params& p)
{
    switch (p.act) {
        case 1: return (void*)bias_act_kernel_hip<T,1>;
        case 2: return (void*)bias_act_kernel_hip<T,2>;
        case 3: return (void*)bias_act_kernel_hip<T,3>;
        case 4: return (void*)bias_act_kernel_hip<T,4>;
        case 5: return (void*)bias_act_kernel_hip<T,5>;
        case 6: return (void*)bias_act_kernel_hip<T,6>;
        case 7: return (void*)bias_act_kernel_hip<T,7>;
        case 8: return (void*)bias_act_kernel_hip<T,8>;
        case 9: return (void*)bias_act_kernel_hip<T,9>;
        default: return nullptr;
    }
}

template void* choose_bias_act_kernel_hip<double>    (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel_hip<float>     (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel_hip<c10::Half> (const bias_act_kernel_params& p);
