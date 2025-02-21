# CUDA-Programming-Project
<p align="justify">
This project explores the Hadamard product, an element-wise multiplication of two matrices, leveraging CUDA for parallel execution on a GPU. Our implementation focuses on optimizing matrix computations using 2D/3D variables with shared memory, a key concept in CUDA programming that significantly enhances performance by reducing global memory access latency.
</p>


## Program Output with Execution Time & Correctness Check 
### C
### CUDA
- Non-Shared, Vector Size (4096x4096), Thread Block (32x32).
![image](https://github.com/user-attachments/assets/cc6ad8df-2c77-4a88-9be0-79c42d73103d)




## Average Execution Time 
![image](https://github.com/user-attachments/assets/31fe8ae7-53c6-4959-874e-c9ef74c0eccb)


## Comparative Analysis

Figure 1.0. Average Kernel Execution Time for Shared 


Figure 2.0. Total Execution Time for Non-Shared
![image](https://github.com/user-attachments/assets/d855551b-aa3e-49be-b8a8-4d0b0ca8984a)


Figure 3.0. Total Execution Time For Shared 

