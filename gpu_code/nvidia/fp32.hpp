#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include "common.hpp"

double sliding_window_fp32(int w = 256, int seq_len = 4096, int head_dim = 64)
{
    int device_id = 0;
    cudaSetDevice(device_id);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float QVec[seq_len][head_dim];
    float KVec[seq_len][head_dim];
    float VVec[seq_len][head_dim];

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < head_dim; j++)
        {
            curandGenerateUniform(gen, &QVec[i][j], 1);
            QVec[i][j] *= 10.0;
            curandGenerateUniform(gen, &KVec[i][j], 1);
            KVec[i][j] *= 10.0;
            curandGenerateUniform(gen, &VVec[i][j], 1);
            VVec[i][j] *= 10.0;
        }
    }

    float *dQ;
    float *dK;
    float *dV;

    cudaMalloc(&dQ, sizeof(float) * seq_len * head_dim);
    cudaMalloc(&dK, sizeof(float) * seq_len * head_dim);
    cudaMalloc(&dV, sizeof(float) * seq_len * head_dim);

    cudaMemcpy(dQ, QVec, sizeof(float) * seq_len * head_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, KVec, sizeof(float) * seq_len * head_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, VVec, sizeof(float) * seq_len * head_dim, cudaMemcpyHostToDevice);

    int num_sliding_chunks = (seq_len - w) / w;
    if (seq_len % w != 0)
        std::cout << "the seq_len is not divisible by w, rounding..." << std::endl;

    int chunk_height = 2 * w;

    float *dS;
    cudaMalloc(&dS, sizeof(float) * num_sliding_chunks * chunk_height * chunk_height);
    cudaMemset(dS, 0, sizeof(float) * num_sliding_chunks * chunk_height * chunk_height);
    float *dZ;
    cudaMalloc(&dZ, sizeof(float) * seq_len * head_dim);
    cudaMemset(dZ, 0, sizeof(float) * seq_len * head_dim);

    float halpha = 1.0f;
    float hbeta = 0.0f;

    double kernel_start = utils_time_us();
    for (int i = 0; i < num_sliding_chunks; i++)
    {
        float *dQ_chunk = dQ + i * w * head_dim;
        float *dK_chunk = dK + i * w * head_dim;
        float *dV_chunk = dV + i * w * head_dim;
        float *dS_chunk = dS + i * chunk_height * chunk_height;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    chunk_height, chunk_height, head_dim, &halpha, dQ_chunk, chunk_height, dK_chunk, head_dim, &hbeta, dS_chunk, chunk_height);
    }

    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N, C, 1));

    for (int i = 0; i < num_sliding_chunks; i++)
    {
        float *S_chunk = dS + i * chunk_height * head_dim;
        float *dV_chunk = dV + i * w * head_dim;
        float *Z_ptr = dZ + i * w * head_dim;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    chunk_height, head_dim, chunk_height, &halpha, S_chunk, head_dim, dV_chunk, chunk_height, &hbeta, Z_ptr, head_dim);
    }
    double kernel_end = utils_time_us();

    // in ms
    double kernel_time = (kernel_end - kernel_start) / 1000.0;

    // Clean up
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dS);
    cudaFree(dZ);
    cublasDestroy(handle);
    curandDestroyGenerator(gen);

    return kernel_time;
}

int main()
{
    double kernel_time = sliding_window_fp32();
    std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;
    return 0;
}