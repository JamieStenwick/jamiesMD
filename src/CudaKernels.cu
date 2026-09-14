#include <cuda/cmath>
#include <chrono>
#include <curanddx.hpp>
#include "Simulator.hpp"
#include "CudaKernels.cuh"


struct ParticleData {
    float* oldPositions {nullptr};
    float* newPositions {nullptr};

    float* oldVelocities {nullptr};
    float* newVelocities {nullptr};

    float* radii {nullptr};
    int* cellIDs {nullptr};

    int N;
};


struct CellData {
    int* cellList {nullptr};
    int* particlesPerCell {nullptr};
    int* cellIndex {nullptr};
    int* cellOffsets {nullptr};
    int* flatNeighborList {nullptr};
};


struct LookupTableData {
    float* forces {nullptr};
    float* energies {nullptr};
    int tableElements;

    float r_min;
    float r_max;
};


struct DeviceParams {
    /*Container for the relevant quantities specifically for integration*/
    DeviceParams(Simulator* Sim) :
    cellsPerSide {Sim->m_Cells.cellsPerSide},
    cellsTotal {cellsPerSide * cellsPerSide * cellsPerSide},
    boxLength {Sim->m_Params.boxLength},
    dt {Sim->m_Params.dt},
    timeSteps {Sim->m_Params.t_steps},
    temp {Sim->m_Params.temp}
    {}

    int cellsPerSide;
    int cellsTotal;
    float boxLength;
    float dt;
    int timeSteps;
    float temp;
};


class DeviceParticles {
    /*Unique owner of the gpu resources for a Particle Data struct
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    ParticleData m_data {};

public:
    explicit DeviceParticles(int N) {
        m_data.N = N;

        cudaMalloc(&m_data.oldPositions, N * 3 * sizeof(float));
        cudaMalloc(&m_data.newPositions, N * 3 * sizeof(float));

        cudaMalloc(&m_data.oldVelocities, N * 3 * sizeof(float));
        cudaMalloc(&m_data.newVelocities, N * 3 * sizeof(float));

        cudaMalloc(&m_data.radii, N * sizeof(float));
        cudaMalloc(&m_data.cellIDs, N * sizeof(float));
    }

    ~DeviceParticles() {
        cudaFree(m_data.oldPositions);
        cudaFree(m_data.newPositions);

        cudaFree(m_data.oldVelocities);
        cudaFree(m_data.newVelocities);

        cudaFree(m_data.radii);
        cudaFree(m_data.cellIDs);
    }

    DeviceParticles(const DeviceParticles&) = delete;
    DeviceParticles& operator=(const DeviceParticles&) = delete;

    ParticleData data() const {
        return m_data;
    }

    void copyFromHost(SimParticles& hostParticles) const {
        /*Copies the particle data owned by a simulator to gpu for integration*/
        cudaMemcpy(m_data.oldPositions, hostParticles.positionsXYZ.data(),
                   m_data.N * 3 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.oldVelocities, hostParticles.velocitiesXYZ.data(),
                   m_data.N * 3 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.radii, hostParticles.radii.data(),
                   m_data.N * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.cellIDs, hostParticles.cellIDs.data(),
                   m_data.N * sizeof(int), cudaMemcpyHostToDevice);
    }
};


class DeviceCells {
    /*Unique owner of the gpu resources for a Cell Data struct
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    CellData m_data {};

public:
    explicit DeviceCells(int N, int cells) {
        cudaMalloc(&m_data.cellList, N * sizeof(int));
        cudaMalloc(&m_data.particlesPerCell, cells * sizeof(int));
        cudaMalloc(&m_data.cellIndex, cells * sizeof(int));
        cudaMalloc(&m_data.cellOffsets, cells * sizeof(int));
        cudaMalloc(&m_data.flatNeighborList, cells * 27 * sizeof(int));
    }

    ~DeviceCells() {
        cudaFree(m_data.cellList);
        cudaFree(m_data.particlesPerCell);
        cudaFree(m_data.cellIndex);
        cudaFree(m_data.cellOffsets);
        cudaFree(m_data.flatNeighborList);
    }

    DeviceCells(const DeviceCells&) = delete;
    DeviceCells& operator=(const DeviceCells&) = delete;

    CellData data() const {
        return m_data;
    }

    // Change to just a reference to SimCell struct when you make it
    void copyFromHost(SimCells hostCells) const {
        /*Copies the cell list owned by a simulator to gpu for integration*/
        cudaMemcpy(m_data.cellList, hostCells.cellList.data(),
                   hostCells.cellList.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.cellIndex, hostCells.cellIndex.data(),
                   hostCells.cellsTotal * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.flatNeighborList, hostCells.flatNeighborList.data(),
                   hostCells.cellsTotal * 27 * sizeof(int), cudaMemcpyHostToDevice);
    }
};


class DeviceLookupTables {
    /*Unique owner of the gpu resources for a LookupTableData struct.
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    LookupTableData m_data {};

public:
    explicit DeviceLookupTables(SimTables& Tables) {
        m_data.tableElements = Tables.N;
        m_data.r_min = Tables.r_min;
        m_data.r_max = Tables.r_max;

        cudaMalloc(&m_data.forces, m_data.tableElements * sizeof(float));
        cudaMalloc(&m_data.energies, m_data.tableElements * sizeof(float));
    }

    ~DeviceLookupTables() {
        cudaFree(m_data.forces);
        cudaFree(m_data.energies);
    }

    DeviceLookupTables(const DeviceLookupTables&) = delete;
    DeviceLookupTables& operator=(const DeviceLookupTables&) = delete;

    LookupTableData data() const {
        return m_data;
    }

    void copyFromHost(SimTables& hostTables) const {
        /*Copies the force tables owned by a simulator to gpu for integration*/
        cudaMemcpy(m_data.forces, hostTables.forceTable.data(),
                   m_data.tableElements * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.energies, hostTables.energyTable.data(),
                   m_data.tableElements * sizeof(float), cudaMemcpyHostToDevice);
    }
};


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


__global__ void integrateKernel(DeviceParticles          Particles,
                                const DeviceLookupTables Tables,
                                DeviceCells              CellData,
                                const DeviceParams       Params,
                                const int**              neighborLists,
                                const int                nPerCell,
                                const int                cellsPerBlock,
                                const unsigned long long seed) {
    /*Assume a maximum of 32 particles per cell (block) for now. In future derive from
    Simulator parameters.*/
    // __shared__ int sharedParticleIDs[nPerCell]; Can't be variable length i guess
    extern __shared__ unsigned char sharedMem[];

    int tid {blockDim.x * blockIdx.x + threadIdx.x};

    // Bounds checking cellIDs
    if (blockIdx.x * cellsPerBlock >= Params.cellsTotal)
        return; // break if we make a persistent kernel
    const int iterations {min(cellsPerBlock, Params.cellsTotal - (blockIdx.x * cellsPerBlock))}; // inline min()

    // First loop over each cellID this block will process
    for (int cell{blockIdx.x * cellsPerBlock}; cell < (blockIdx.x * cellsPerBlock) + iterations; ++cell) {
        
        // Get individual particleID for this thread, using threadIdx to bound when threads > particles
        // And load initial cell positions in here and figure out how to loop over all particles j != i

        // Then loop over each neighbor cell
        for (int neighborIdx{cell * 27}; neighborIdx < cell * 27 + 27; ++neighborIdx) {


        }
    }

}


void integrateKernelWrapper(Simulator* Sim) {
    /*Takes a pointer to a Simulator object, allocates device memory for integration,
    and orchestrates Simulator host actions and Integrator Device actions during the
    duration of the integration. Right now integrates from start to finish.*/

    // Random number generation setup
    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>()
                   + curanddx::PhiloxRounds<6>()
                   + curanddx::SM<860>()
                   + curanddx::Thread());
    const unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Allocating GPU memory
    DeviceParticles Particles{Sim->m_Params.N};
    DeviceCells Cells{Sim->m_Params.N, Sim->m_Cells.cellsTotal};
    DeviceLookupTables Tables{Sim->m_Tables};
    DeviceParams Params{Sim};

    // Copying initial values from CPU to GPU
    Particles.copyFromHost(Sim->m_Particles);
    Cells.copyFromHost(Sim->m_Cells);
    Tables.copyFromHost(Sim->m_Tables);

    // Launch Kernel
    constexpr int particlesPerCell {32}; // Derive from SimParams in future (nearest multiple of 32)
    constexpr int blocks {100}; // Derive from memory limit in future, i.e. the maximum number you can have simultaneously while accounting for thread limits per SM, register limits, and number of SM's

    // Number of cells each block needs to do a full computation over every particle in said cells,
    // in other words: total iterations = 27 * iterationsPerBlock
    const int cellsPerBlock {(Sim->m_Cells.cellsTotal / blocks) + 1};

    // PASS THE AMOUNT OF MEMORY ALLOCATED AS 3RD KERNEL ARGUMENT
    // integrateKernel<<<blocks, particlesPerCell, sharedBytes>>>
}
