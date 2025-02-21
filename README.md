# CUDA-Programming-Project
<p align="justify">
This project explores the Hadamard product, an element-wise multiplication of two matrices, leveraging CUDA for parallel execution on a GPU. Our implementation focuses on optimizing matrix computations using 2D/3D variables with shared memory.
</p>


## Program Output with Execution Time & Correctness Check 
### A. C Program
- C program, Vector size is 4096x4096
![image](https://github.com/user-attachments/assets/b51a50e4-cb93-427a-b3e5-092291f81006)

### B. CUDA Program 
- Non-Shared, Vector Size (4096x4096), Thread Block (32x32).
![image](https://github.com/user-attachments/assets/cc6ad8df-2c77-4a88-9be0-79c42d73103d)

![Screenshot_20250221_202208](https://github.com/user-attachments/assets/af359fc9-83c9-45fc-b6bc-b5664f9fbd4d)

- Shared, Vector Size (4096x4096), Thread Block (32x32).

![Screenshot_20250221_201926](https://github.com/user-attachments/assets/d509352b-42f1-4ab9-9c1a-a81e6208452a)

![Screenshot_20250221_202059](https://github.com/user-attachments/assets/5b9bc016-a1b9-4d13-93af-f40c084ea22d)



## Comparative Analysis

- Figure 1.0. Average Kernel Execution Time for Shared 
![image](https://github.com/user-attachments/assets/05f942e1-4ad5-4670-a872-7d6244136bab)



- Figure 2.0. Total Execution Time for Non-Shared  
![Screenshot_20250221_201603](https://github.com/user-attachments/assets/54643eb2-1d41-47d0-903f-82f19f5177e1)


- Figure 3.0. Total Execution Time For Shared 

![Screenshot_20250221_201656](https://github.com/user-attachments/assets/94e6e770-661c-437e-9ce6-26f954f347b6)

<p align="justify">
From a glance, there is a clear performance difference between the non-shared and shared memory implementations as seen from both figures 2 and 3. Knowing the inceot of shared memory using global memory, the shared memory approach benefits from **improved memory coalescing, reduced global memory traffic, and efficient data reusability.** In the non-shared version, every thread must individually fetch data from global memory, this is a process that can quickly become a bottleneck as the matrix size increases. With shared memory, however, data is loaded once per block and then reused by all threads in that block, drastically reducing the number of slow global memory accesses.
  
</p>

<p align="justify">
Furthermore it is observed that the thread block configuration is crucial for maximizing performance. Optimal block sizes enhance GPU occupancy and ensure that the limited shared memory is used efficiently. While larger blocks may lead to improved parallelism and lower overhead, they can also strain shared memory resources if not properly managed. Balancing these factors is key to achieving the best performance. Overall, the shared memory strategy not only minimizes latency but also leverages on-chip data reuse to deliver significant speedups, making it a highly effective optimization for parallel operations like the Hadamard product.
</p>





## Links 
1. Youtube Link
2. Additional Resources (Docs): https://docs.google.com/document/d/1QE3GzmY2PIk5JYr5eSgXj4tQDgYsYG_G-rJZWC_iVQA/edit


