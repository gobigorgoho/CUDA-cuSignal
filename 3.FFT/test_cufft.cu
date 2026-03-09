#include <iostream>
#include <vector>
#include <iomanip>
#include <cufft.h>
#include "common.cuh"
#include "complex_math.cuh"

int main() {
    setGPU(0);
    
    // 设置信号长度 (必须是能进行 FFT 的长度，通常是 2 的幂次方最快)
    const int N = 8;
    size_t bytes = N * sizeof(cuFloatComplex);

   // 引入 PI 常量 
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    // 1. 初始化主机端数据 (构造混合频率信号)
    std::vector<cuFloatComplex> h_signal(N);
    for (int i = 0; i < N; ++i) {
        // 信号 1：频率 bin 为 1，幅度为 1.0
        float theta1 = 2.0f * M_PI * 1 * i / N;
        // 信号 2：频率 bin 为 3，幅度为 0.5
        float theta2 = 2.0f * M_PI * 3 * i / N;

        // 根据欧拉公式 e^(j*theta) = cos(theta) + j*sin(theta) 构造复数
        float real_part = 1.0f * cos(theta1) + 0.5f * cos(theta2);
        float imag_part = 1.0f * sin(theta1) + 0.5f * sin(theta2);

        h_signal[i] = make_complex(real_part, imag_part);
    }
    std::vector<cuFloatComplex> h_fft_out(N);
    std::vector<cuFloatComplex> h_ifft_out(N);

    // 2. 分配设备端内存 (注意：cuFloatComplex 和 cufftComplex 可以直接强转)
    cufftComplex *d_signal, *d_fft_out, *d_ifft_out;
    CHECK_CUDA(cudaMalloc((void**)&d_signal, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_fft_out, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_ifft_out, bytes));

    // 数据搬运 H2D
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal.data(), bytes, cudaMemcpyHostToDevice));

    // ---------------------------------------------------------
    // 3. cuFFT 核心流程三部曲：建计划 -> 执行 -> 销毁
    // ---------------------------------------------------------
    
    // a. 创建 FFT 计划 (Plan)
    cufftHandle plan;
    // 1D FFT，长度为 N，类型为 CUFFT_C2C (复数到复数)
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1)); 

    // b. 执行正向 FFT (CUFFT_FORWARD)
    std::cout << ">>> Executing Forward FFT..." << std::endl;
    CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_fft_out, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize()); // 确保计算完成

    // c. 执行逆向 IFFT (CUFFT_INVERSE)
    std::cout << ">>> Executing Inverse FFT..." << std::endl;
    // 直接把频域结果 d_fft_out 作为输入，反变换到 d_ifft_out
    CHECK_CUFFT(cufftExecC2C(plan, d_fft_out, d_ifft_out, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // d. 销毁计划，释放资源
    CHECK_CUFFT(cufftDestroy(plan));

    // ---------------------------------------------------------

    // 4. 将结果拷回主机 D2H
    CHECK_CUDA(cudaMemcpy(h_fft_out.data(), d_fft_out, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ifft_out.data(), d_ifft_out, bytes, cudaMemcpyDeviceToHost));

    // 5. 打印验证
    std::cout << "\n--- cuFFT 验证结果 (N=" << N << ") ---\n";
    std::cout << "Idx | Original (Time)   | FFT Output (Freq)   | IFFT Output (Time, Unscaled)\n";
    std::cout << "----------------------------------------------------------------------------\n";
    for (int i = 0; i < N; ++i) {
        std::cout << i << "   | " 
                  << std::setw(5) << h_signal[i].x << "+" << h_signal[i].y << "i | "
                  << std::setw(5) << h_fft_out[i].x << "+" << h_fft_out[i].y << "i | "
                  << std::setw(5) << h_ifft_out[i].x << "+" << h_ifft_out[i].y << "i\n";
    }

    // 释放设备内存
    cudaFree(d_signal); cudaFree(d_fft_out); cudaFree(d_ifft_out);
    return 0;
}