#include <iostream>
#include <vector>
#include <cufft.h>
#include <math_constants.h>
#include "common.cuh"
#include "complex_math.cuh"

// 1. 生成 Hamming 窗核函数 (参考 cusignal/windows/windows.py)
// 公式: w(n) = 0.54 - 0.46 * cos(2*pi*n / (M-1))
__global__ void generate_hamming_kernel(float* window, int M) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < M) {
        window[n] = 0.54f - 0.46f * cosf(2.0f * CUDART_PI_F * n / (M - 1));
    }
}

// 2. 分帧加窗 Kernel: 将 1D 信号填入 2D 矩阵的每一行，并乘以窗函数
__global__ void stft_framing_kernel(
    const cuFloatComplex* input, 
    cuFloatComplex* output_frames, 
    const float* window,
    int nperseg,    // 每一帧长度 (窗口大小)
    int noverlap,   // 重叠长度
    int nstep,      // 步长 (nperseg - noverlap)
    int n_frames,   // 总帧数
    int signal_len) 
{
    int frame_idx = blockIdx.y; // 哪一帧
    int sample_idx = threadIdx.x; // 帧内第几个采样点

    if (frame_idx < n_frames && sample_idx < nperseg) {
        int input_start = frame_idx * nstep;
        int input_idx = input_start + sample_idx;

        int out_idx = frame_idx * nperseg + sample_idx;

        if (input_idx < signal_len) {
            // 信号值 * 窗函数值
            float w = window[sample_idx];
            output_frames[out_idx] = make_complex(input[input_idx].x * w, input[input_idx].y * w);
        } else {
            output_frames[out_idx] = make_complex(0.0f, 0.0f); // 补零
        }
    }
}

int main() {
    setGPU(0);

    // 参数设置
    const int signal_len = 4096;   // 总信号长度
    const int nperseg = 256;       // 窗口大小 (FFT长度)
    const int noverlap = 128;      // 重叠一半
    const int nstep = nperseg - noverlap;
    const int n_frames = (signal_len - noverlap) / nstep;

    // 1. 模拟一个频率震荡的信号 (微多普勒特征)
    std::vector<cuFloatComplex> h_input(signal_len);
    for(int i=0; i<signal_len; ++i) {
        // 基础频率 + 随时间震荡的频率
        float freq = 10.0f + 5.0f * sinf(2.0f * M_PI * i / 1000.0f);
        float phase = 2.0f * M_PI * freq * i / 1024.0f;
        h_input[i] = make_complex(cosf(phase), sinf(phase));
    }

    // 分配内存
    cuFloatComplex *d_input, *d_frames, *d_spec;
    float *d_window;
    CHECK_CUDA(cudaMalloc(&d_input, signal_len * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_frames, n_frames * nperseg * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_spec, n_frames * nperseg * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMalloc(&d_window, nperseg * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), signal_len * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // 2. 执行流水线
    // A. 生成窗函数
    generate_hamming_kernel<<<(nperseg+255)/256, 256>>>(d_window, nperseg);

    // B. 分帧加窗
    dim3 grid(1, n_frames);
    dim3 block(nperseg);
    stft_framing_kernel<<<grid, block>>>(d_input, d_frames, d_window, nperseg, noverlap, nstep, n_frames, signal_len);

    // C. 批量 FFT (核心复用)
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, nperseg, CUFFT_C2C, n_frames));
    CHECK_CUFFT(cufftExecC2C(plan, d_frames, d_spec, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. 结果分析 (打印前几帧的中心频率位置)
    std::vector<cuFloatComplex> h_spec(n_frames * nperseg);
    CHECK_CUDA(cudaMemcpy(h_spec.data(), d_spec, n_frames * nperseg * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    std::cout << ">>> STFT 时频分析完成！" << std::endl;
    std::cout << "帧序号 | 峰值频率位置 (体现频率随时间变化)" << std::endl;
    for (int f = 0; f < 10; ++f) {
        float max_mag = 0;
        int max_bin = 0;
        for (int b = 0; b < nperseg; ++b) {
            float mag = abs_val(h_spec[f * nperseg + b]);
            if (mag > max_mag) { max_mag = mag; max_bin = b; }
        }
        std::cout << "Frame " << f << " | Peak Bin: " << max_bin << std::endl;
    }

    cufftDestroy(plan);
    // 释放内存...
    return 0;
}