#include <iostream>
#include <vector>
#include <iomanip>
#include "bsplines.cuh"

int main() {
    // 初始化GPU
    setGPU(0);
    // 设置测试规模
    const int N = 5;
    size_t bytes = N * sizeof(float);

    // 准备主机端测试数据
    // 特意选几个边界点和中心点来测试样条逻辑
    // vector自动调用析构函数释放内存
    std::vector<float> h_x = {-1.5f, -0.5f, 0.0f, 0.5f, 1.5f};
    std::vector<float> h_y(N, 0.0f);

    std::cout << "Input x array: ";
    // 遍历h_x数组中的每一个元素，把它们依次打印
    for (float val : h_x) std::cout << val << " ";
    std::cout << "\n\n";

    // 分配设备端显存
    float *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_x, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_y, bytes));

    // 将数据从主机拷贝到GPU
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    // 测试Cubic B-spline
    std::cout << "--- Testing Cubic B-spline ---" << std::endl;
    cubic_spline<float>(d_x, d_y, N);
    CHECK_CUDA(cudaDeviceSynchronize()); // 确保GPU计算完毕
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {
        std::cout << "x: " << std::setw(4) << h_x[i] << " -> res: " << h_y[i] << std::endl;
    }

    // 测试: Quadratic B-spline
    std::cout << "\n--- Testing Quadratic B-spline ---" << std::endl;
    quadratic_spline<float>(d_x, d_y, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {
        std::cout << "x: " << std::setw(4) << h_x[i] << " -> res: " << h_y[i] << std::endl;
    }

    //测试 gauss_spline when n = 3
    std::cout << "\n--- Testing gauss B-spline ---" << std::endl;
    gauss_spline<float>(d_x, d_y, 3, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        std::cout << "x: " << std::setw(4) << h_x[i] << " -> res: " << h_y[i] << std::endl;
    }

    // 释放显存
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}