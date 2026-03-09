#include <iostream>
#include <vector>
#include <iomanip>
#include "common.cuh"          
#include "complex_math.cuh"    // 引入我们的复数数学库

// 【核心核函数】模拟脉冲压缩中的频域相乘： Output = Signal * Conj(Template)
__global__ void pulse_compression_math_kernel(
    const cuFloatComplex* signal, 
    const cuFloatComplex* chirp_template, 
    cuFloatComplex* output, 
    float* output_magnitude, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 1. 取出当前点的数据
        cuFloatComplex s = signal[idx];
        cuFloatComplex t = chirp_template[idx];
        
        // 2. 核心数学运算：因为我们重载了运算符，这里可以直接写 * 和 conj()！
        cuFloatComplex res = s * conj(t);
        
        // 3. 顺便算一下模长（用于最终在屏幕上显示亮度）
        float mag = abs_val(res);
        
        // 4. 写回全局内存
        output[idx] = res;
        output_magnitude[idx] = mag;
    }
}

int main() {
    setGPU(0); // 初始化 GPU
    const int N = 5;
    size_t bytes = N * sizeof(cuFloatComplex);

    // 1. 在主机端 (CPU) 构造一些测试的 I/Q 复数数据
    std::vector<cuFloatComplex> h_signal = {
        make_complex(1.0f, 1.0f), make_complex(0.0f, 2.0f), 
        make_complex(-1.0f, 0.0f), make_complex(3.0f, -1.0f), make_complex(0.5f, 0.5f)
    };
    std::vector<cuFloatComplex> h_template = {
        make_complex(1.0f, -1.0f), make_complex(0.0f, 1.0f), 
        make_complex(1.0f, 1.0f), make_complex(-1.0f, 2.0f), make_complex(1.0f, 0.0f)
    };
    
    std::vector<cuFloatComplex> h_output(N);
    std::vector<float> h_magnitude(N);

    // 2. 分配设备端 (GPU) 内存
    cuFloatComplex *d_signal, *d_template, *d_output;
    float *d_magnitude;
    CHECK_CUDA(cudaMalloc(&d_signal, bytes));
    CHECK_CUDA(cudaMalloc(&d_template, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_magnitude, N * sizeof(float)));

    // 3. 数据搬运 H2D (Host to Device)
    // 思考点：cuFloatComplex 底层是 float2，内存是交错排列的 (Real, Imag, Real, Imag...)。
    // 这种排列在 DMA 传输时天然具有很好的连续性，能把总线带宽跑满。
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_template, h_template.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    pulse_compression_math_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_signal, d_template, d_output, d_magnitude, N
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. 结果取回 D2H (Device to Host)
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_magnitude.data(), d_magnitude, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. 打印验证
    std::cout << "--- 模拟脉冲压缩复数运算结果 ---" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Sig: (" << h_signal[i].x << "," << h_signal[i].y << ") * "
                  << "Conj_Tpl: (" << h_template[i].x << "," << -h_template[i].y << ") = "
                  << "Out: (" << std::setw(4) << h_output[i].x << "," << std::setw(4) << h_output[i].y << ")"
                  << " | Mag: " << h_magnitude[i] << std::endl;
    }

    // 释放内存
    cudaFree(d_signal); cudaFree(d_template); cudaFree(d_output); cudaFree(d_magnitude);
    return 0;
}