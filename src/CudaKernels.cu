#include <cuda/cmath>
#include <curanddx.hpp>

using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() // type of generator, 4 at a time with this
                   + curanddx::PhiloxRounds<6>() // number of bijections, 6 is lowest (larger the better I think)
                   + curanddx::SM<860>() // specific architecture setting
                   + curanddx::Thread()); // initializes state on thread

float4 generate4();

__global__ void preComputeRandomNumbersInitial() {

}


void preComputeWrapperInitial() {
    // constants (to be moved to function arguments)
    const int N {8000};

    // Allocate CPU memory, put all 3 coordinates in one array for cache optimization
    float* hostPositions {nullptr};

    float* hostVelocites {nullptr};

    cudaMallocHost(&hostX, N * sizeof(float));
    cudaMallocHost(&hostY, N * sizeof(float));
    cudaMallocHost(&hostZ, N * sizeof(float));

    cudaMallocHost(&hostVx, N * sizeof(float));
    cudaMallocHost(&hostVy, N * sizeof(float));
    cudaMallocHost(&hostVz, N * sizeof(float));

    // Allocate GPU memory
    float* devX {nullptr};
    float* devY {nullptr};
    float* devZ {nullptr};

    float* devVx {nullptr};
    float* devVy {nullptr};
    float* devVz {nullptr};


}
