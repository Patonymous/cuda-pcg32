#pragma once

#include <exception>
#include <string>

#include <cuda_runtime.h>

static cudaError_t cudaErrno;

class message_exception : public std::exception
{
private:
    std::string message;
public:
    inline message_exception() : message() { }
    inline message_exception(const char* msg) : message(msg) { }
    inline message_exception(std::string msg) : message(msg) { }
    virtual const char *what() const noexcept override { return message.c_str(); }
};

#define CUDA_THROW_ON_ERROR(action) { if ((cudaErrno = action) != cudaSuccess) throw message_exception(cudaGetErrorString(cudaErrno)); }
#define CTE(action) CUDA_THROW_ON_ERROR(action)

#define CUDA_RETURN_ON_ERROR(action) { if ((cudaErrno = action) != cudaSuccess) return cudaErrno; }
#define CRE(action) CUDA_RETURN_ON_ERROR(action)
