//cgeist gemm.cu -function=matmul -S --resource-dir=$LLVM_BUILD_DIR/lib/clang/18 --cuda-gpu-arch=sm_75
//--cuda-path=/usr/local/cuda-11.8 -emit-cuda
//set env var
//export LLVM_BUILD_DIR=/mnt/d/LLVM/NewPolygeistDir/llvm-project/build
#include <cuda_runtime.h>

#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

__global__ void matmul_kernel(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        DATA_TYPE sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

void matmul(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    DATA_TYPE *d_A, *d_B, *d_C;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, N * K * sizeof(DATA_TYPE));
    cudaMalloc((void **)&d_B, K * M * sizeof(DATA_TYPE));
    cudaMalloc((void **)&d_C, N * M * sizeof(DATA_TYPE));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * K * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * M * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
