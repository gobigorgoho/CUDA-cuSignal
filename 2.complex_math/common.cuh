#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

inline void setGPU(int device_id=0)
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&iDeviceCount));

    if (iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    if (device_id >= iDeviceCount)
    {
        std::cerr << "Warning: 请求的 GPU ID (" << device_id 
                  << ") 超出范围，强制使用 0 号设备。" << std::endl;
        device_id = 0;
    }
    
    // 设置执行
    CHECK_CUDA(cudaSetDevice(device_id));
    std::cout << "  GPU 初始化成功" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // GPU热身
    // 强制 CUDA 运行时立刻建立Context，消除冷启动延迟
    CHECK_CUDA(cudaFree(0));
}
