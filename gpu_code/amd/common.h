#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <vector>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <rocsparse.h>
#include <miopen/miopen.h>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if (stat != hipSuccess)                                                \
        {                                                                      \
            std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
            exit(-1);                                                          \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if (stat != rocsparse_status_success)                                        \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }

#define ROCBLAS_CHECK(stat)                                                          \
    {                                                                                \
        if (stat != rocsparse_status_success)                                        \
        {                                                                            \
            std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
            exit(-1);                                                                \
        }                                                                            \
    }

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if (status != rocblas_status_success)             \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

inline double utils_time_us(void)
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

#endif // COMMON_H