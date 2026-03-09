#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

// 核心 API 声明
// 计算高斯样条
template <typename T>
void gauss_spline(const T* d_x, T* d_output, int n, int size);

// 计算三次 B-样条
template <typename T>
void cubic_spline(const T* d_x, T* d_output, int size);

// 计算二次 B-样条
template <typename T>
void quadratic_spline(const T* d_x, T* d_output, int size);