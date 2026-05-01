#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                              \
    do {                                                              \
        cudaError_t _e = (expr);                                      \
        if (_e != cudaSuccess)                                        \
            throw std::runtime_error(std::string("CUDA error: ") +   \
                                     cudaGetErrorString(_e) +         \
                                     " at " __FILE__ ":" +            \
                                     std::to_string(__LINE__));       \
    } while (0)
