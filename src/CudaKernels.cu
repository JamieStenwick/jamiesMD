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
    float* oldForces {nullptr};

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
    float stepSize;
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
        cudaMalloc(&m_data.oldForces, N * 3 * sizeof(float));

        cudaMalloc(&m_data.radii, N * sizeof(float));
        cudaMalloc(&m_data.cellIDs, N * sizeof(float));
    }

    ~DeviceParticles() {
        cudaFree(m_data.oldPositions);
        cudaFree(m_data.newPositions);

        cudaFree(m_data.oldVelocities);
        cudaFree(m_data.newVelocities);
        cudaFree(m_data.oldForces);

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
        cudaMalloc(&m_data.cellIndex, cells * sizeof(int) + 1);
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
        m_data.stepSize = (m_data.r_max - m_data.r_min) / m_data.tableElements;

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


__device__ __forceinline__ float3 operator+(const float3& arr1, const float3& arr2) {
    return {arr1.x + arr2.x, arr1.y + arr2.y, arr1.z + arr2.z};
}


__device__ __forceinline__ float3 operator*(const float3& arr, const float scalar) {
    return {arr.x * scalar, arr.y * scalar, arr.z * scalar};
}


// Put this in a device friendly format
__device__ __forceinline__ float4 getDistance(const float3& pos1, const float3& pos2, const float boxLength) {
    /*Takes two particles' positions in float3, and returns a float4 with the .x .y .z .w
    attributes equal to dx, dy, dz, and dr respectively. Handles periodic boundary conditions.*/
    float dx {pos1.x - pos2.x};
    dx -= boxLength * round(dx / boxLength);

    float dy {pos1.y - pos2.y};
    dy -= boxLength * round(dy / boxLength);

    float dz {pos1.z - pos2.z};
    dz -= boxLength * round(dz / boxLength);

    return {dx, dy, dz, dr};
}


__global__ void forceSum(ParticleData          Particles,
                         const CellData        Cells,
                         const LookupTableData LookupTables,
                         const DeviceParams    Params) {
    /*Assume a maximum of 32 particles per cell (block) for now. In future derive from
    Simulator parameters. Also, threads that don't own a particle in the center cell
    are effectively doing nothing for that entire iteration, even when there are more
    particles in a neighbor cell than there were in the center.*/

    __shared__ float sharedPos[3 * 32];

    // Bounds checking cellIDs
    const int cellsPerBlock {cuda::ceil_div(Params.cellsTotal, gridDim.x)};
    if (blockIdx.x * cellsPerBlock >= Params.cellsTotal)
        return;

    // First loop over each center cellID this block will process
    const int iterations {min(cellsPerBlock, Params.cellsTotal - (blockIdx.x * cellsPerBlock))}; // inline min()
    for (int cellID{blockIdx.x * cellsPerBlock}; cellID < (blockIdx.x * cellsPerBlock) + iterations; ++cellID) {
        
        __syncthreads();
        // Center cell case is special because we need to loop over every particle j != i

        // Bounding threads to number of particles in cell
        const int particlesPerCell {CellData.particlesPerCell[cellID]};
        if (threadIdx.x >= particlesPerCell)
            continue;

        // Particle i assigned to this thread and loading that position into shared mem
        const int particleID {CellData.cellList[CellData.cellIndex[cellID] + threadIdx.x]};
        sharedPos[threadIdx.x] = Particles.oldPositions[particleID];
        sharedPos[threadIdx.x + particlesPerCell] = Particles.oldPositions[particleID + Params.N];
        sharedPos[threadIdx.x + 2 * particlesPerCell] = Particles.oldPositions[particleID + Params.N];

        // Loop over every particle j != i
        float3 force{0, 0, 0};
        for (int offset{1}; offset < particlesPerCell; ++offset) {
            const int jthIdx {(offset + threadIdx.x) % particlesPerCell};

            const float4 distance {getDistance({sharedPos[threadIdx.x], sharedPos[threadIdx.x + particlesPerCell], sharedPos[threadIdx.x + 2*particlesPerCell]},
                                               {sharedPos[jthIdx], sharedPos[jthIdx + particlesPerCell], sharedPos[jthIdx + 2*particlesPerCell]})};

            if (distance.w <= LookupTables.r_max && distance.w >= LookupTables.r_min) {
                const float forceMagnitude {LookupTables.forces[round((distance.w - LookupTables.r_min) / LookupTables.stepSize)]};
                force.x += forceMagnitude * distance.x / distance.w;
                force.y += forceMagnitude * distance.y / distance.w;
                force.z += forceMagnitude * distance.z / distance.w;
            }
        }

        // Then loop over each neighbor cell
        for (int neighborIdx{cellID * 27}; neighborIdx < cellID * 27 + 27; ++neighborIdx) {


        }
    }

}




}


__device__ inline float3 BAOA(int particleID, float3 velocity, float3 randomForce, ParticleData Particles, DeviceParams Params) {
    /*Takes particleID to read last time step's force and position from VRAM,
    Takes velocity from register and updates it, and takes ptr to sharedPos to immediately
    write calculated positions into.*/
    const float3 force {Particles.forces[particleID],
                        Particles.forces[particleID + Particles.N],
                        Particles.forces[particleID + 2*Particles.N]};

    float3 position {Particles.oldPositions[particleID],
                 Particles.oldPositions[particleID + Particles.N],
                 Particles.oldPositions[particleID + 2*Particles.N]};

    velocity += force * (Params.dt / 2); // B, mass=1 in denominator
    position += velocity * (Params.dt / 2); // A

    const float c {expf(-Params.dt)}; // drag coef gamma=1 and mass=1, both in exponent
    const float sigma {sqrtf(1.38 * Params.temp * (1 - c)*(1 - c))} // kT will likely non-dimensionalize
    velocity = (velocity * c) + (randomForce * sigma); // O

    position += velocity * (Params.dt / 2); // A

    // Writing updated positions to global in preparation for force sum step
    Particles.newPositions[threadIdx.x] = position.x;
    Particles.newPositions[threadIdx.x + blockDim.x] = position.y;
    Particles.newPositions[threadIdx.x + 2*blockDim.x] = position.z;

    return velocity;
}


// It makes way more sense to just fucking separate the force sum and BAOA integration, so that's what
// we're going to do, remember: programming is like destiny 2 day one raiding, do the simplest possible thing
// first to get the clear, then later you can optimize individual modules (operations) and overall design (algorithmic)
// both are easier after having already programmed the whole thing
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
    // integrateKernel<<<blocks, particlesPerCell, sharedBytes>>>(Particles.data(), Cells.data(), Tables.data(), Params,)
}
