import numpy as np

# 1. 构造与 C++ 完全一致的输入数据
signal = np.array([1+1j, 0+2j, -1+0j, 3-1j, 0.5+0.5j], dtype=np.complex64)
template = np.array([1-1j, 0+1j, 1+1j, -1+2j, 1+0j], dtype=np.complex64)

# 2. 一行代码完成 numpy 端的计算 (Python 这里的便捷正是我们要用 CUDA 去硬核实现的)
output = signal * np.conj(template)
magnitude = np.abs(output)

# 3. 打印结果，用于和 C++ 终端输出进行肉眼比对
print("--- Python Numpy 验证结果 ---")
for i in range(len(signal)):
    print(f"Index {i}: Out: ({output[i].real}, {output[i].imag}) | Mag: {magnitude[i]:.6f}")