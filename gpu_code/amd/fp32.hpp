#include "common.h"
double sliding_window_fp32(int w = 256, int seq_len = 4096, int head_dim = 64)
{
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    float QVec[seq_len][head_dim];
    float KVec[seq_len][head_dim];
    float VVec[seq_len][head_dim];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < head_dim; j++)
        {
            QVec[i][j] = dis(gen);
            KVec[i][j] = dis(gen);
            VVec[i][j] = dis(gen);
        }
    }

    float *dQ;
    float *dK;
    float *dV;

    HIP_CHECK(hipMalloc(&dQ, sizeof(float) * seq_len * head_dim));
    HIP_CHECK(hipMalloc(&dK, sizeof(float) * seq_len * head_dim));
    HIP_CHECK(hipMalloc(&dV, sizeof(float) * seq_len * head_dim));

    HIP_CHECK(hipMemcpy(dQ, QVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, KVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, VVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));

    int num_sliding_chunks = (seq_len - w) / w;
    if (seq_len % w != 0)
        std::cout << "the seq_len is not divisible by w, arounding..." << std::endl;
    // std::cout << "num_sliding_chunks: " << num_sliding_chunks << std::endl;
    int chunk_height = 2 * w;

    float *dS;
    HIP_CHECK(hipMalloc(&dS, sizeof(float) * num_sliding_chunks * chunk_height * chunk_height));
    HIP_CHECK(hipMemset(dS, 0, sizeof(float) * num_sliding_chunks * chunk_height * chunk_height));
    float *dZ;
    HIP_CHECK(hipMalloc(&dZ, sizeof(float) * seq_len * head_dim));
    HIP_CHECK(hipMemset(dZ, 0, sizeof(float) * seq_len * head_dim));

    float halpha = static_cast<float>(1);
    float hbeta = static_cast<float>(0);

    double kernel_start = utils_time_us();
    for (int i = 0; i < num_sliding_chunks; i++)
    {
        float *dQ_chunk = dQ + i * w * head_dim;
        float *dK_chunk = dK + i * w * head_dim;
        float *dV_chunk = dV + i * w * head_dim;
        float *dS_chunk = dS + i * chunk_height * chunk_height;
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      chunk_height, chunk_height, head_dim, &halpha, dQ_chunk, chunk_height, dK_chunk, head_dim, &hbeta, dS_chunk, chunk_height);
    }
    miopenTensorDescriptor_t inputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenSet4dTensorDescriptor(inputDesc, miopenFloat, 1, num_sliding_chunks, chunk_height, chunk_height);
    miopenHandle_t miopen_handle;
    miopenCreate(&miopen_handle);
    miopenSoftmaxForward(miopen_handle, &halpha, inputDesc, dS, &hbeta, inputDesc, dS);
    for (int i = 0; i < num_sliding_chunks; i++)
    {
        float *S_chunk = dS + i * chunk_height * head_dim;
        float *dV_chunk = dV + i * w * head_dim;
        float *Z_ptr = dZ + i * w * head_dim;
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      chunk_height, head_dim, chunk_height, &halpha, S_chunk, head_dim, dV_chunk, chunk_height, &hbeta, Z_ptr, head_dim);
    }
    double kernel_end = utils_time_us();

    // in ms
    double kernel_time = (kernel_end - kernel_start) / 1000.0;
    return kernel_time;
};

double sddmm_fp32(int w = 256, int seq_len = 4096, int head_dim = 64)
{
    float halpha = static_cast<float>(1);
    float hbeta = static_cast<float>(0);

    // random input
    float QVec[seq_len][head_dim];
    float KVec[seq_len][head_dim];
    float VVec[seq_len][head_dim];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < head_dim; j++)
        {
            QVec[i][j] = dis(gen);
            KVec[i][j] = dis(gen);
            VVec[i][j] = dis(gen);
        }
    }

    // generating the sparse matrix
    std::vector<float> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    row_ptr.push_back(0);
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = i - w; j < i + w; j++)
        {
            if (j >= 0 && j < seq_len)
            {
                values.push_back(1.0);
                col_idx.push_back(j);
            }
        }
        row_ptr.push_back(values.size());
    }
    int nnz = values.size();

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    hipDeviceProp_t devProp;
    int device_id = 0;

    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    float *dQ;
    float *dK;
    float *dV;
    HIP_CHECK(hipMalloc(&dQ, sizeof(float) * seq_len * head_dim));
    HIP_CHECK(hipMalloc(&dK, sizeof(float) * seq_len * head_dim));
    HIP_CHECK(hipMalloc(&dV, sizeof(float) * seq_len * head_dim));
    // DEBUG
    std::cout << "22" << std::endl;
    HIP_CHECK(hipMemcpy(dQ, QVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));
    // DEBUG
    std::cout << "33" << std::endl;
    HIP_CHECK(hipMemcpy(dK, KVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, VVec, sizeof(float) * seq_len * head_dim, hipMemcpyHostToDevice));

    HIP_CHECK(hipDeviceSynchronize());

    // DEBUG
    std::cout << "22" << std::endl;

    int *dCptr;
    int *dCcol;
    float *dCval;
    HIP_CHECK(hipMalloc((void **)&dCptr, sizeof(int) * (head_dim + 1)));
    HIP_CHECK(hipMalloc((void **)&dCcol, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void **)&dCval, sizeof(float) * nnz));
    HIP_CHECK(hipMemcpy(dCptr, row_ptr.data(), sizeof(int) * (head_dim + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCcol, col_idx.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCval, values.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_spmat_descr M;
    rocsparse_dnmat_descr Q;
    rocsparse_dnmat_descr K;
    rocsparse_dnmat_descr V;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&M, seq_len, seq_len, nnz, dCptr, dCcol, dCval, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&Q, seq_len, head_dim, seq_len, dQ, rocsparse_datatype_f32_r, rocsparse_order_row));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&K, head_dim, seq_len, head_dim, dK, rocsparse_datatype_f32_r, rocsparse_order_row));
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&V, seq_len, head_dim, seq_len, dV, rocsparse_datatype_f32_r, rocsparse_order_row));

    size_t buffer_size;
    double kernel_start = utils_time_us();
    ROCSPARSE_CHECK(rocsparse_sddmm(handle,
                                    rocsparse_operation_none,
                                    rocsparse_operation_none,
                                    &halpha,
                                    Q,
                                    K,
                                    &hbeta,
                                    M,
                                    rocsparse_datatype_f32_r,
                                    rocsparse_sddmm_alg_default,
                                    nullptr));
    double kernel_end = utils_time_us();
    return (kernel_end - kernel_start) / 1000.0;
}