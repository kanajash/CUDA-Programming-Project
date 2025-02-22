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
From a glance, there is a clear performance difference between the non-shared and shared memory implementations as seen from both figures 2 and 3. Knowing the inceot of shared memory using global memory, the shared memory approach benefits from improved memory coalescing, reduced global memory traffic, and efficient data reusability. In the non-shared version, every thread must individually fetch data from global memory, this is a process that can quickly become a bottleneck as the matrix size increases. With shared memory, however, data is loaded once per block and then reused by all threads in that block, drastically reducing the number of slow global memory accesses.
  
</p>

<p align="justify">
Furthermore it is observed that the thread block configuration is crucial for maximizing performance. Optimal block sizes enhance GPU occupancy and ensure that the limited shared memory is used efficiently. While larger blocks may lead to improved parallelism and lower overhead, they can also strain shared memory resources if not properly managed. Comparing different execution times from differect vector sizes and thread block sizes help us unserdtand the importance of picking the right size and knowing how the process works. 

For example on the vector size of 1024, the smallest size among the three, the 8x8 thread block size is the fastest withe the time of 26,200ns. There is also an obervation that the increase number of threads in the context of larger blocks did not significantly affect the performance of smaller matrix sizes  this maybe due to higher thread overhead and synchronization under smaller workloads 


in the size of 2048, there is an upward trend in execution times. the larger blocks handles the larger matrices efficiently than the smaller value, meaning as the block size increases it reduces the execution time gradually. This implies that the mid-range matrix sizes has better parallelism and has better memory access patterns 



Therefore, smaller thread blocks perform better with small matrices (1024) , while larger blocks are favorable for the medium sized matrices such as 2048. For a larger matrix such as 4096, the thread block size does not significantly affect the execution time. 

  
This just suggests that it should be considered that in impementing these configurations, not only the thread block size is to be thought out but also the compatible matrix size. Balancing these factors is key to achieving the best performance. Overall, the shared memory strategy not only minimizes latency but also leverages on-chip data reuse to deliver significant speedups, making it a highly effective optimization for parallel operations like the Hadamard product.
</p>


One of the problem encountered when implementing the shared memory for the CUDA kernel was that it ran slower than the non-shared version which should not be the case because according to Nvidia's shared memory article, a shared memory implementation should run much faster than it non-shared counterpart. A possible explanation for this is the bank conflict, it is when a half warp (16 threads) tries to load/store data from or to the same bank. At first the implementation to store an element to a 2D array was `sh_X[threadIdx.x][threadIdx.y]` which at first it made sense from the perspective of accessing a 2d array with the format of `arr[row][col]`. Since in a thread the fastest varying or the the one that changes a lot as index increments is x (row), instead of the y (column) with the table below we can see as to why `sh_X[threadIdx.x][threadIdx.y]` indexing schemee would cause the bank conflict, since increments will vary first with `threadIdx.x` therefore incrementing its row first hitting multiple elements in bank0 until half warp so changing the 2D array indexing scheme to `sh_X[threadIdx.y][threadIdx.x]` removed the bank conflict.

![Screenshot_20250221_234816](https://github.com/user-attachments/assets/e5171cf4-d789-420b-ba02-1261c8bd2079)

Table from: https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming


## Links 
1. Youtube Link: https://youtu.be/xAKhL31pI6c
2. Additional Resources (Docs): https://docs.google.com/document/d/1QE3GzmY2PIk5JYr5eSgXj4tQDgYsYG_G-rJZWC_iVQA/edit
3. Shared Memory: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
4. https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming


