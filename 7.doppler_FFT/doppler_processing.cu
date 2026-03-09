#include <iostream>
#include <vector>
#include <iomanip>
#include <math_constants.h>
#include <cufft.h>
#include "common.cuh"
#include "complex_math.cuh"

// 1. 生成带有多普勒频移的 2D 回波矩阵 (128行 x 1024列)
__global__ void generate_2d_echo_kernel(
    cuFloatComplex* signal_out, 
    float fs, float T, float B, 
    int target_range, int target_doppler, 
    int num_pulses, int N) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 距离维 (快时间)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 脉冲维 (慢时间)

    if (col < N && row < num_pulses) {
        int idx = row * N + col; // 二维矩阵的 1D 索引
        
        if (col >= target_range) {
            float t = (col - target_range) / fs;
            if (t <= T) {
                float K = B / T;
                float phase_chirp = CUDART_PI_F * K * t * t;
                // 多普勒相位旋转：随着脉冲序号(row)的增加，相位发生有规律的旋转
                float phase_doppler = 2.0f * CUDART_PI_F * target_doppler * row / num_pulses;
                
                float total_phase = phase_chirp + phase_doppler;
                signal_out[idx] = make_complex(cosf(total_phase), sinf(total_phase));
            } else {
                signal_out[idx] = make_complex(0.0f, 0.0f);
            }
        } else {
            signal_out[idx] = make_complex(0.0f, 0.0f);
        }
    }
}

// 2. 二维批量频域相乘 (所有的行都乘以同一个发射波形模板)
__global__ void batched_freq_multiply_kernel(
    const cuFloatComplex* sig_freq, // 2D 矩阵
    const cuFloatComplex* tmp_freq, // 1D 模板
    cuFloatComplex* out_freq, 
    int num_pulses, int N) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < num_pulses) {
        int idx = row * N + col;
        out_freq[idx] = sig_freq[idx] * conj(tmp_freq[col]);
    }
}

// 3. 缩放与求模长，生成最终的 RD 图
__global__ void magnitude_2d_kernel(
    const cuFloatComplex* rd_matrix, 
    float* magnitude, 
    int num_pulses, int N) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < num_pulses) {
        int idx = row * N + col;
        // 经历了两次 FFT (距离向和方位向)，所以除以 (N * num_pulses)
        float scale = 1.0f / (N * num_pulses);
        cuFloatComplex scaled = make_complex(rd_matrix[idx].x * scale, rd_matrix[idx].y * scale);
        magnitude[idx] = abs_val(scaled);
    }
}

// 复用上一个任务的 Chirp 生成核函数 (仅生成 1D 模板)
__global__ void generate_chirp_kernel(cuFloatComplex* template_out, float fs, float T, float B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = idx / fs;
        if (t < T) {
            float phase = CUDART_PI_F * (B / T) * t * t; 
            template_out[idx] = make_complex(cosf(phase), sinf(phase));
        } else {
            template_out[idx] = make_complex(0.0f, 0.0f);
        }
    }
}

int main() {
    setGPU(0);
    
    // --- 物理参数 ---
    const float fs = 1e6f;  
    const float T  = 64e-6f;
    const float B  = 0.5e6f;
    
    // --- 矩阵维度 ---
    const int N = 1024;           // 列数：距离采样点
    const int NUM_PULSES = 128;   // 行数：脉冲个数
    const int TOTAL_ELEMENTS = N * NUM_PULSES;
    
    // --- 模拟目标参数 ---
    const int TARGET_RANGE = 200;   // 模拟目标距离 Bin
    const int TARGET_DOPPLER = 15;  // 模拟目标速度 Bin (多普勒频移)

    // 分配显存
    cufftComplex *d_signal_2d, *d_template_1d;
    cufftComplex *d_sig_freq_2d, *d_tmp_freq_1d, *d_pc_out_2d, *d_rd_map_2d;
    float *d_magnitude_2d;

    CHECK_CUDA(cudaMalloc((void**)&d_signal_2d, TOTAL_ELEMENTS * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_sig_freq_2d, TOTAL_ELEMENTS * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_pc_out_2d, TOTAL_ELEMENTS * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_rd_map_2d, TOTAL_ELEMENTS * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_magnitude_2d, TOTAL_ELEMENTS * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc((void**)&d_template_1d, N * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp_freq_1d, N * sizeof(cuFloatComplex)));

    // --- 1. 生成数据 ---
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (NUM_PULSES + threads.y - 1) / threads.y);
    
    generate_chirp_kernel<<<(N + 255)/256, 256>>>(d_template_1d, fs, T, B, N);
    generate_2d_echo_kernel<<<blocks, threads>>>(d_signal_2d, fs, T, B, TARGET_RANGE, TARGET_DOPPLER, NUM_PULSES, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // =========================================================
    // 核心流水线 A：距离向脉冲压缩 (Batched 1D FFT over Rows)
    // =========================================================
    
    // 1. 专门为 1D 发射模板创建一个 FFT Plan (batch = 1)
    cufftHandle plan_template;
    CHECK_CUFFT(cufftPlan1d(&plan_template, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan_template, d_template_1d, d_tmp_freq_1d, CUFFT_FORWARD));

    // 2. 为 2D 回波矩阵创建一个 Batched FFT Plan (batch = 128)
    cufftHandle plan_range;
    CHECK_CUFFT(cufftPlan1d(&plan_range, N, CUFFT_C2C, NUM_PULSES));
    CHECK_CUFFT(cufftExecC2C(plan_range, d_signal_2d, d_sig_freq_2d, CUFFT_FORWARD));
    
    // 3. 频域相乘 (2D * 1D)
    batched_freq_multiply_kernel<<<blocks, threads>>>(d_sig_freq_2d, d_tmp_freq_1d, d_pc_out_2d, NUM_PULSES, N);
    
    // 4. 逆变换回时域
    CHECK_CUFFT(cufftExecC2C(plan_range, d_pc_out_2d, d_pc_out_2d, CUFFT_INVERSE));

   // =========================================================
    // 核心流水线 B：多普勒处理 (Strided 1D FFT over Columns)
    // =========================================================
    cufftHandle plan_doppler;
    int n[1] = {NUM_PULSES};
    
    // 修复：将原本的 NULL 替换为 n，激活 GPU 的高级跳跃访存模式！
    CHECK_CUFFT(cufftPlanMany(&plan_doppler, 1, n, 
                              n, N, 1,   // 替换 NULL 为 n
                              n, N, 1,   // 替换 NULL 为 n
                              CUFFT_C2C, N));

    CHECK_CUFFT(cufftExecC2C(plan_doppler, d_pc_out_2d, d_rd_map_2d, CUFFT_FORWARD));

    // 生成最后的 2D 能量图
    magnitude_2d_kernel<<<blocks, threads>>>(d_rd_map_2d, d_magnitude_2d, NUM_PULSES, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- 取回结果并全图搜索最高峰 ---
    std::vector<float> h_magnitude_2d(TOTAL_ELEMENTS);
    CHECK_CUDA(cudaMemcpy(h_magnitude_2d.data(), d_magnitude_2d, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    float max_val = 0.0f;
    int found_range = -1, found_doppler = -1;
    
    for (int r = 0; r < NUM_PULSES; ++r) {
        for (int c = 0; c < N; ++c) {
            float val = h_magnitude_2d[r * N + c];
            if (val > max_val) {
                max_val = val;
                found_range = c;
                found_doppler = r;
            }
        }
    }

    std::cout << ">>> 2D Range-Doppler 矩阵处理完成！" << std::endl;
    std::cout << "设定目标 -> 距离 (Range): " << TARGET_RANGE << ", 速度 (Doppler): " << TARGET_DOPPLER << std::endl;
    std::cout << "探测目标 -> 距离 (Range): " << found_range << ", 速度 (Doppler): " << found_doppler << std::endl;
    std::cout << "峰值能量: " << max_val << std::endl;

    cufftDestroy(plan_template);
    cufftDestroy(plan_range); 
    cufftDestroy(plan_doppler);
    return 0;
}