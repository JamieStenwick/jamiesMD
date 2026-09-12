#include <cuda/cmath>
#include <chrono>
#include <curanddx.hpp>


template<class RNG>
__global__ void populateRandomKernel(float devOut[],
                                      const unsigned long long seed,
                                      const std::size_t N,
                                      const float boxLength) {
    /*Takes pointer to GPU output array, the seed, length of the output, and
    the bounds of the distribution to sample*/
    const std::size_t i {blockDim.x * blockIdx.x + threadIdx.x};
    const std::size_t j {3 * i};

    // Bounds checking make sure our indices are valid
    if (j >= N)
        return;
    
    // Subsequence and offset are ways to index into the RNG sequence.
    // Both subsequence and offset are 64-bit, so really big
    curanddx::uniform<float> uniformDist(-boxLength / 2, +boxLength / 2);
    RNG rng(seed, i, 0);
    float4 rand {uniformDist.generate4(rng)};

    devOut[j] = rand.x;
    devOut[j + 1] = rand.y;
    devOut[j + 2] = rand.z;
}


void popRandKernelWrap(float hostOut[], const std::size_t N, const float boxLength) {
    /*Takes a pointer to an array on the host CPU to copy results to, the size of the array,
    and the bounds and generates the requisite number of floats in a uniform distribution in
    the bounds. Will need to accomadate normal distribution for velocities at some point.*/
    // Allocate GPU memory
    float* devOut {nullptr};
    cudaMalloc(&devOut, N * sizeof(float));

    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() // type of generator, 4 at a time with this
                   + curanddx::PhiloxRounds<10>() // number of bijections, 6 is lowest (larger the better I think)
                   + curanddx::SM<860>() // specific architecture setting
                   + curanddx::Thread()); // initializes state on thread

    // Seed with system clock
    const unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Invoke kernel
    const std::size_t N_particles = N / 3;
    const int threads {256};
    const int blocks {static_cast<int>(cuda::ceil_div(N_particles, static_cast<std::size_t>(threads)))}; // 3 numbers, 1 particle per thread
    populateRandomKernel<RNG><<<blocks, threads>>>(devOut, seed, N, boxLength);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(hostOut, devOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(devOut);
}
