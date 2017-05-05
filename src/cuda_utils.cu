#include "cuda_utils.h"

/*
__device__ inline float atomicCAS(float* address, float compare, float val) {
    auto r = atomicCAS((int *)address, __float_as_int(compare), __float_as_int(val));
    return __int_as_float(r);
}*/

void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}




