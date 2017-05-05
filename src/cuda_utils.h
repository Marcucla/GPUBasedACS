#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);


__device__ inline float atomicCAS(float* address, float compare, float val) {
    auto r = atomicCAS((int *)address, __float_as_int(compare), __float_as_int(val));
    return __int_as_float(r);
}


template<typename T>
class DeviceVector {
public:

    DeviceVector() :
        size_(0),
        dev_ptr_(nullptr),
        id_("") {
    }

    DeviceVector(size_t size, const std::string &id = "") :
        size_(size),
        dev_ptr_(nullptr) {
        gpuErrchk( cudaMalloc((void **)&dev_ptr_, size * sizeof(T)) );
        id_ = id;
    }

    DeviceVector(size_t size, int8_t value, const std::string &id = "") :
        size_(size),
        dev_ptr_(nullptr) {
        gpuErrchk( cudaMalloc((void **)&dev_ptr_, size * sizeof(T)) );
        gpuErrchk( cudaMemset((void **)&dev_ptr_, size * sizeof(T), value) );
        id_ = id;
    }

    DeviceVector(const std::vector<T> &vec, const std::string &id = "") :
        size_(vec.size()) {
        gpuErrchk( cudaMalloc((void **)&dev_ptr_, size_ * sizeof(T)) );
        copyFrom(vec);
        id_ = id;
    }

    template<typename R>
    DeviceVector(const std::vector<std::vector<R>> &matrix, 
                 const std::string &id = "") 
        : id_(id) {
        
        size_t len = 0;
        for (auto &row : matrix) {
            len += row.size();
        }

        std::vector<T> buffer;
        buffer.reserve(len);
        for (auto &row : matrix) {
            for (auto el : row) {
                buffer.push_back( (T)el );
            }
        }

        size_ = len;
        gpuErrchk( cudaMalloc((void **)&dev_ptr_, size_ * sizeof(T)) );
        copyFrom(buffer);
    }

    virtual ~DeviceVector() {
        if (dev_ptr_) {
            gpuErrchk( cudaFree(dev_ptr_) );
            dev_ptr_ = nullptr;
        }
    }


    void resize(size_t size) {
        if (size_ != size) {
            size_ = size;
            if (dev_ptr_) {
                gpuErrchk( cudaFree(dev_ptr_) );
            }
            gpuErrchk( cudaMalloc((void **)&dev_ptr_, size_ * sizeof(T)) );
        }
    }


    T* data() const { return dev_ptr_; }

    size_t size() const { return size_; }

    void copyFrom(const std::vector<T> &vec) {
        gpuErrchk( cudaMemcpy(dev_ptr_, vec.data(),
                    std::min(size_, vec.size()) * sizeof(T),
                    cudaMemcpyHostToDevice) );
        // gpuErrchk( cudaDeviceSynchronize() );
    }

    void copyTo(std::vector<T> &vec) {
        vec.resize(size_);
        gpuErrchk( cudaMemcpy(&vec.front(), dev_ptr_,
                              size_ * sizeof(T),
                              cudaMemcpyDeviceToHost) );
    }

    void copyTo(T *vec) {
        gpuErrchk( cudaMemcpy(vec, dev_ptr_,
                              size_ * sizeof(T),
                              cudaMemcpyDeviceToHost) );
    }

private:
    std::string id_;
    size_t size_;
    T *dev_ptr_; 
};


/*
  Broadcast val from leader thread to other warps
 */ 
template<class T>
__device__ __forceinline__
T warp_bcast(T val, int leader) { 
    return __shfl(val, leader); 
}

 
template<class T, int warp_size>
__device__ __forceinline__
T warp_scan(T val) {
    T my_sum = val;
    for (int i = 1; i < warp_size; i <<= 1) {
        const T other = __shfl_up(my_sum, i);
        if (threadIdx.x >= i) my_sum += other;
    }
    return my_sum;
}


template<typename T, int warp_size>
__device__ __forceinline__
T warp_reduce(T val) {
    T my_sum = val;
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        my_sum += __shfl_down(my_sum, offset);
    }
    return my_sum;
}


template<typename arg_t, typename value_t, int warp_size>
__device__ __forceinline__
arg_t warp_reduce_arg_max(arg_t arg, value_t val) {
    value_t my_max = val;
    // Find nn node with max total value using warp level shuffling
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        const value_t other_max = __shfl_down(my_max, offset);
        const arg_t other_arg = __shfl_down(arg, offset);
        arg = my_max < other_max ? other_arg : arg; 
        my_max = max(my_max, other_max);
    }
    return arg;
}



template<typename arg_t, typename value_t, int warp_size>
__device__ __forceinline__
arg_t warp_reduce_arg_max_ext(arg_t arg, value_t val, float *out_max) {
    value_t my_max = val;
    // Find nn node with max total value using warp level shuffling
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        const value_t other_max = __shfl_down(my_max, offset);
        const arg_t other_arg = __shfl_down(arg, offset);
        arg = my_max < other_max ? other_arg : arg; 
        my_max = max(my_max, other_max);
    }
    *out_max = my_max;
    return arg;
}

/* 

Utility class to simplify creating objects in CUDA unified memory.

See: http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/

*/
class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    printf("Her");
    gpuErrchk( cudaMallocManaged(&ptr, len) );
    return ptr;
  }

  void operator delete(void *ptr) {
    gpuErrchk( cudaFree(ptr) );
  }
};


/* 
 * Global alias for GPU side random number generator state
 */
//typedef curandStatePhilox4_32_10_t gpu_rand_state_t;
typedef curandStateXORWOW gpu_rand_state_t;


struct PRNG {
    gpu_rand_state_t *global_state_;
    gpu_rand_state_t state_; // Cache global PRNG state for faster generation


    __device__ __forceinline__ PRNG(gpu_rand_state_t *global_state) 
        : global_state_(global_state),
          state_(*global_state)
    {}


    __device__ ~PRNG() {
        // On destroy update global state
        *global_state_ = state_;
    }

    __device__ __forceinline__ float rand_uniform() {
        return curand_uniform(&state_);
    }

    __device__ __forceinline__ uint32_t rand_uint() {
        return (uint32_t)curand(&state_);
    }
};



struct GPUIntervalTimer {

    GPUIntervalTimer() {
        gpuErrchk( cudaEventCreate(&start_) );
        gpuErrchk( cudaEventCreate(&stop_) );
    }

    ~GPUIntervalTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start_interval() {
        gpuErrchk( cudaEventRecord(start_) );
    }

    void stop_interval() {
        gpuErrchk( cudaEventRecord(stop_) );
        gpuErrchk( cudaEventSynchronize(stop_) );
        float ms = 0; 
        cudaEventElapsedTime(&ms, start_, stop_);
        total_ += ms;
    }

    void reset() { 
        total_ = 0;
    }

    /*
     * Returns total time in seconds
     */
    double get_total_time() const { return total_ / 1.0e3; }

    double get_total_time_ms() const { return total_; }

    double total_ = 0.0;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

#endif
