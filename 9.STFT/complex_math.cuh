#pragma once
#include <cuComplex.h>
#include <math.h>

// 这是一个极其重要的技巧：重载运算符。
// 加上 __device__ __forceinline__ 可以让这些小函数在编译时直接展开在核函数里，完全没有函数调用的开销。

// 1. 复数加法 (a + b)
__host__ __device__ __forceinline__ cuFloatComplex operator+(const cuFloatComplex& a, const cuFloatComplex& b) {
    return cuCaddf(a, b);
}

// 2. 复数减法 (a - b)
__host__ __device__ __forceinline__ cuFloatComplex operator-(const cuFloatComplex& a, const cuFloatComplex& b) {
    return cuCsubf(a, b);
}

// 3. 复数乘法 (a * b) - 这是脉冲压缩最核心的运算
__host__ __device__ __forceinline__ cuFloatComplex operator*(const cuFloatComplex& a, const cuFloatComplex& b) {
    return cuCmulf(a, b);
}

// 4. 求共轭 (Conjugate) - 脉冲压缩需要用到发射信号的共轭
__host__ __device__ __forceinline__ cuFloatComplex conj(const cuFloatComplex& a) {
    return cuConjf(a);
}

// 5. 求模长 (Magnitude/Absolute value) - 多普勒处理完后，需要求模长才能画 RD 图
__host__ __device__ __forceinline__ float abs_val(const cuFloatComplex& a) {
    return cuCabsf(a);
}

// 6. 辅助函数：用实部和虚部构造复数
__host__ __device__ __host__ __forceinline__ cuFloatComplex make_complex(float r, float i) {
    return make_cuFloatComplex(r, i);
}