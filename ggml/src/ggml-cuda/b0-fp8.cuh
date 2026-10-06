#pragma once
#include "common.cuh"
#include <cfloat>

static __device__ __forceinline__ float b0_div_rn(float a, float b) {
    float result;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    result = a/b;
#else
    // No .ftz and no approximate reciprocal from --use_fast_math.
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
#endif
    return result;
}
static __device__ __forceinline__ float b0_bf16_round(float v) {
    uint32_t bits = __float_as_uint(v);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(bits);
}
static __device__ __forceinline__ uint8_t b0_encode_e4m3(float v) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const __nv_fp8_e4m3 value(v);
    return value.__x;
#else
    const uint8_t sign = __float_as_uint(v) >> 31 ? 128 : 0;
    const float mag = fabsf(v);
    int low = 0, high = 126;
    while (high-low > 1) {
        int mid = (low+high)/2;
        if (ggml_cuda_e4m3_to_fp32(uint8_t(mid)) <= mag) low = mid; else high = mid;
    }
    const float dl = mag-ggml_cuda_e4m3_to_fp32(uint8_t(low));
    const float dh = ggml_cuda_e4m3_to_fp32(uint8_t(high))-mag;
    const int code = dl < dh ? low : dh < dl ? high : (low & 1) ? high : low;
    return uint8_t(code) | sign;
#endif
}
static __device__ __forceinline__ void quantize_f32_b0_fp8_mla_block(const float * x, block_b0_fp8_mla * out) {
    // Validate the whole row before writing, including overflow in BF16 rounding.
    for (int j = 0; j < 512; ++j) {
        if (!isfinite(x[j]) || !isfinite(b0_bf16_round(x[j]))) __trap();
    }
    for (int g = 0; g < 4; ++g) {
        uint32_t max_bits = 0;
        for (int j = 0; j < 128; ++j) {
            const uint32_t bits = __float_as_uint(b0_bf16_round(x[g*128+j])) & 0x7fffffffu;
            max_bits = max_bits > bits ? max_bits : bits;
        }
        const float scale = fmaxf(b0_div_rn(__uint_as_float(max_bits), 448.0f), FLT_MIN);
        out->scales[g] = scale;
        for (int j = 0; j < 128; ++j) {
            const float v = b0_div_rn(b0_bf16_round(x[g*128+j]), scale);
            out->qs[g*128+j] = b0_encode_e4m3(fminf(fmaxf(v, -448.0f), 448.0f));
        }
    }
    for (int j = 0; j < 64; ++j) out->rope[j] = 0;
}
static __device__ __forceinline__ void dequantize_b0_fp8_mla(const void * ptr, const int64_t block, const int index, float2 & result) {
    const block_b0_fp8_mla & x = ((const block_b0_fp8_mla *) ptr)[block];
    const float scale = x.scales[index/128];
    if (scale == 0) {
        const uint8_t * bytes = (const uint8_t *) &x;
        for (int j = 0; j < 656; ++j) if (bytes[j] != 0) __trap();
    } else if (!isfinite(scale) || scale < FLT_MIN) __trap();
    if ((x.qs[index] & 127) == 127 || (x.qs[index+1] & 127) == 127) __trap();
    const float a = ggml_cuda_e4m3_to_fp32(x.qs[index]);
    const float b = ggml_cuda_e4m3_to_fp32(x.qs[index+1]);
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    result.x = a*scale;
    result.y = b*scale;
#else
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result.x) : "f"(a), "f"(scale));
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result.y) : "f"(b), "f"(scale));
#endif
}
