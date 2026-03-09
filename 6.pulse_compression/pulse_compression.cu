#include <iostream>
#include <vector>
#include <iomanip>
#include <cufft.h>
#include "common.cuh"
#include "complex_math.cuh"

// 1. 频域相乘核函数： Output_Freq = Signal_Freq * Conj(Template_Freq)
__global__ void freq_multiply_kernel(
    const cuFloatComplex* sig_freq, 
    const cuFloatComplex* tmp_freq, 
    cuFloatComplex* out_freq, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 复用我们在 complex_math.cuh 中重载的运算符 * 和 conj()
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
        // IFFT 结果除以 N
        cuFloatComplex scaled = make_complex(ifft_out[idx].x / size, ifft_out[idx].y / size);
        
        // 求模长 (用于判断能量峰值)
        magnitude[idx] = abs_val(scaled);
    }
}

int main() {
    setGPU(0);
    
    // 设定信号总长度 (比如 1024 个采样点)
    const int N = 1024;
    size_t bytes = N * sizeof(cuFloatComplex);
    
    // 模拟目标位置：假设目标在距离 Index = 150 的地方
    const int TARGET_INDEX = 150;
    const int PULSE_WIDTH = 64; // 发射脉冲的宽度

    // 1. 初始化主机端数据
    std::vector<cuFloatComplex> h_template(N, make_complex(0.0f, 0.0f));
    std::vector<cuFloatComplex> h_signal(N, make_complex(0.0f, 0.0f));
    std::vector<float> h_magnitude(N, 0.0f);

    // 构造发射波形 (简单起见，这里用一个长度为 64 的方波脉冲代替 Chirp)
    for (int i = 0; i < PULSE_WIDTH; ++i) {
        h_template[i] = make_complex(1.0f, 0.0f);
    }

    // 构造接收回波：将波形移动到 TARGET_INDEX 处，模拟目标回波
    for (int i = 0; i < PULSE_WIDTH; ++i) {
        h_signal[TARGET_INDEX + i] = make_complex(1.0f, 0.0f);
    }

    // 2. 分配设备端内存
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

    // 数据搬运 H2D
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_template, h_template.data(), bytes, cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // 3. 脉冲压缩核心流水线
    // ---------------------------------------------------------
    
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Step A: 分别对回波和发射波形做正向 FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_sig_freq, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_template, d_tmp_freq, CUFFT_FORWARD));
    
    // Step B: 频域相乘 (Signal_Freq * Conj(Template_Freq))
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    freq_multiply_kernel<<<blocks, threads>>>(d_sig_freq, d_tmp_freq, d_mult_freq, N);
    CHECK_CUDA(cudaGetLastError());

    // Step C: 逆变换 IFFT 回到时域
    CHECK_CUFFT(cufftExecC2C(plan, d_mult_freq, d_ifft_out, CUFFT_INVERSE));
    
    // Step D: 幅度缩放与求模长
    scale_and_magnitude_kernel<<<blocks, threads>>>(d_ifft_out, d_magnitude, N);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftDestroy(plan));

    // ---------------------------------------------------------
    // 4. 拷回结果并寻找目标峰值
    CHECK_CUDA(cudaMemcpy(h_magnitude.data(), d_magnitude, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << ">>> 脉冲压缩完成！正在寻找目标位置..." << std::endl;
    
    float max_val = 0.0f;
    int target_idx = -1;
    for (int i = 0; i < N; ++i) {
        if (h_magnitude[i] > max_val) {
            max_val = h_magnitude[i];
            target_idx = i;
        }
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "设置的模拟目标位置: Index = " << TARGET_INDEX << std::endl;
    std::cout << "雷达探测到的目标位置: Index = " << target_idx << std::endl;
    std::cout << "该点峰值能量 (Magnitude): " << max_val << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 释放内存
    cudaFree(d_signal); cudaFree(d_template);
    cudaFree(d_sig_freq); cudaFree(d_tmp_freq); 
    cudaFree(d_mult_freq); cudaFree(d_ifft_out); cudaFree(d_magnitude);
    return 0;
}