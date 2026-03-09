#include <iostream>
#include <vector>
#include <iomanip>
#include <math_constants.h> // CUDA 数学常量库，提供 CUDART_PI_F (即 pi)
#include <cufft.h>
#include "common.cuh"
#include "complex_math.cuh"

// --- 新增：Chirp 发射波形生成 Kernel ---
__global__ void generate_chirp_kernel(
    cuFloatComplex* template_out, 
    float fs, float T, float B, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = idx / fs; // 当前点的物理时间
        if (t <= T) {
            float K = B / T; // 调频斜率
            // 计算相位: phi = pi * K * t^2
            float phase = CUDART_PI_F * K * t * t; 
            
            // 欧拉公式生成复数: cos(phi) + j*sin(phi)
            // 注意：设备端强制使用单精度函数 cosf 和 sinf 以获得最佳性能
            template_out[idx] = make_complex(cosf(phase), sinf(phase));
        } else {
            // 脉冲结束后的时间填 0
            template_out[idx] = make_complex(0.0f, 0.0f);
        }
    }
}

// --- 新增：模拟接收回波生成 Kernel (带距离延迟) ---
__global__ void generate_echo_kernel(
    cuFloatComplex* signal_out, 
    float fs, float T, float B, int target_idx, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (idx >= target_idx) {
            float t = (idx - target_idx) / fs; // 减去延迟，对齐脉冲起点
            if (t <= T) {
                float K = B / T;
                float phase = CUDART_PI_F * K * t * t;
                signal_out[idx] = make_complex(cosf(phase), sinf(phase));
            } else {
                signal_out[idx] = make_complex(0.0f, 0.0f);
            }
        } else {
            // 信号到来前全是 0 (真实场景这里应填入环境噪声)
            signal_out[idx] = make_complex(0.0f, 0.0f);
        }
    }
}

// 1. 频域相乘核函数： Output_Freq = Signal_Freq * Conj(Template_Freq)
__global__ void freq_multiply_kernel(
    const cuFloatComplex* sig_freq, 
    const cuFloatComplex* tmp_freq, 
    cuFloatComplex* out_freq, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out_freq[idx] = sig_freq[idx] * conj(tmp_freq[idx]);
    }
}

// 2. IFFT 后处理核函数：除以 N (幅度缩放) 并求模长
__global__ void scale_and_magnitude_kernel(
    const cuFloatComplex* ifft_out, 
    float* magnitude, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cuFloatComplex scaled = make_complex(ifft_out[idx].x / size, ifft_out[idx].y / size);
        magnitude[idx] = abs_val(scaled);
    }
}

int main() {
    setGPU(0);
    
    // ---------------------------------------------------------
    // 雷达物理参数配置
    // ---------------------------------------------------------
    const float fs = 1e6f;       // 采样率：1 MHz
    const float T  = 64e-6f;     // 脉冲宽度：64 微秒 (占 64 个采样点)
    const float B  = 0.5e6f;     // 信号带宽：500 kHz (带宽越大，距离分辨率越高)
    
    const int N = 1024;          // 接收窗总长度
    const int TARGET_INDEX = 200;// 模拟目标距离位置
    size_t bytes = N * sizeof(cuFloatComplex);

    // 分配设备端内存
    cufftComplex *d_signal, *d_template;
    cufftComplex *d_sig_freq, *d_tmp_freq, *d_mult_freq, *d_ifft_out;
    float *d_magnitude;

    CHECK_CUDA(cudaMalloc((void**)&d_signal, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_template, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_sig_freq, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp_freq, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_mult_freq, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_ifft_out, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_magnitude, N * sizeof(float)));

    // ---------------------------------------------------------
    // 小任务 5：完全在 GPU 上生成 Chirp 波形
    // ---------------------------------------------------------
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    generate_chirp_kernel<<<blocks, threads>>>(d_template, fs, T, B, N);
    generate_echo_kernel<<<blocks, threads>>>(d_signal, fs, T, B, TARGET_INDEX, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------------------------------------------------------
    // 脉冲压缩核心流水线
    // ---------------------------------------------------------
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Step A: 正向 FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_sig_freq, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_template, d_tmp_freq, CUFFT_FORWARD));
    
    // Step B: 频域相乘
    freq_multiply_kernel<<<blocks, threads>>>(d_sig_freq, d_tmp_freq, d_mult_freq, N);
    
    // Step C: 逆向 IFFT
    CHECK_CUFFT(cufftExecC2C(plan, d_mult_freq, d_ifft_out, CUFFT_INVERSE));
    
    // Step D: 缩放与求模
    scale_and_magnitude_kernel<<<blocks, threads>>>(d_ifft_out, d_magnitude, N);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftDestroy(plan));

    // ---------------------------------------------------------
    // 取回并分析结果
    // ---------------------------------------------------------
    std::vector<float> h_magnitude(N);
    CHECK_CUDA(cudaMemcpy(h_magnitude.data(), d_magnitude, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << ">>> Chirp 脉冲压缩完成！" << std::endl;
    std::cout << "目标设定位置: Index = " << TARGET_INDEX << std::endl;
    
    // 我们不仅打印最高点，还要打印最高点附近的值，让你亲眼看看“一根针”是什么样
    std::cout << "\n观察目标点附近的能量分布 (雷达的距离分辨率体现):" << std::endl;
    for (int i = TARGET_INDEX - 3; i <= TARGET_INDEX + 3; ++i) {
        std::cout << "Index " << i << " -> Magnitude: " << std::fixed << std::setprecision(2) << h_magnitude[i];
        if (i == TARGET_INDEX) std::cout << "  <-- 绝对的尖峰！";
        std::cout << std::endl;
    }

    // 释放内存
    cudaFree(d_signal); cudaFree(d_template);
    cudaFree(d_sig_freq); cudaFree(d_tmp_freq); 
    cudaFree(d_mult_freq); cudaFree(d_ifft_out); cudaFree(d_magnitude);
    return 0;
}