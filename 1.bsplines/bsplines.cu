#include "bsplines.cuh"
#include <cmath>

// 定义数学常量（若未定义）
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Kernels (设备端)
// gauss_spline_kernel
// 核函数处理并行数据，通常以指针形式传入数据的首地址
// T可在调用时指定
template <typename T>
__global__ void gauss_spline_kernel(const T* x, T* output, double signsq, double r_signsq, int size) {
    //一维网格一维线程块
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (T)(1.0 / sqrt(2.0 * M_PI * signsq) * exp(-(x[idx] * x[idx]) * r_signsq));
    }
}

//cubic_kernel(n=3)
template <typename T>
__global__ void cubic_kernel(const T* x, T* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const T ax = abs(x[idx]);
        if (ax < 1) {
            res[idx] = 2.0 / 3 - 1.0 / 2 * ax * ax * (2.0 - ax);
        } else if (!(ax < 1) && (ax < 2)) {
            res[idx] = 1.0 / 6 * (2.0 - ax) * (2.0 - ax) * (2.0 - ax);
        } else {
            res[idx] = 0.0;
        }
    }
}

//quadratic_kernel
template <typename T>
__global__ void quadratic_kernel(const T* x, T* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const T ax = abs(x[idx]);
        if (ax < 0.5) {
            res[idx] = 0.75 - ax * ax;
        } else if (!(ax < 0.5) && (ax < 1.5)) {
            res[idx] = ((ax - 1.5) * (ax - 1.5)) * 0.5;
        } else {
            res[idx] = 0.0;
        }
    }
}

// Host (主机端)
template <typename T>
void gauss_spline(const T* d_x, T* d_output, int n, int size) {
    // 提前在 CPU 计算标量常量
    double signsq = (n + 1) / 12.0;
    double r_signsq = 0.5 / signsq;
    
    const int threadsPerBlock = 256;
    dim3 block(threadsPerBlock);
    dim3 grid((size + block.x - 1) / block.x);
    
    //调用核函数
    gauss_spline_kernel<T><<<grid, block>>>(d_x, d_output, signsq, r_signsq, size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void cubic_spline(const T* d_x, T* d_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cubic_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_x, d_output, size);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void quadratic_spline(const T* d_x, T* d_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    quadratic_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_x, d_output, size);
    CHECK_CUDA(cudaGetLastError());
}

// 显式模板实例化（为了让main.cu可以链接到这些模板函数）
template void gauss_spline<float>(const float* d_x, float* d_output, int n, int size);
template void gauss_spline<double>(const double* d_x, double* d_output, int n, int size);

template void cubic_spline<float>(const float* d_x, float* d_output, int size);
template void cubic_spline<double>(const double* d_x, double* d_output, int size);

template void quadratic_spline<float>(const float* d_x, float* d_output, int size);
template void quadratic_spline<double>(const double* d_x, double* d_output, int size);