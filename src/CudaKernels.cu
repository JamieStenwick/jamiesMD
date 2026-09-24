#include <cuda/cmath>
#include <chrono>
#include <curanddx.hpp>
#include <cub/device/device_scan.cuh>
#include "Simulator.hpp"
#include "CudaKernels.cuh"
#include <iostream>


struct ParticleData {
    float* positions {nullptr};
    float* velocities {nullptr};
    float* forces {nullptr};

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
    float* forceEnergyTable {nullptr};
    int tableElements;

    float r_min;
    float r_max;
    float maxForce;
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

        cudaMalloc(&m_data.positions, N * 3 * sizeof(float));
        cudaMalloc(&m_data.velocities, N * 3 * sizeof(float));
        cudaMalloc(&m_data.forces, N * 3 * sizeof(float));

        cudaMalloc(&m_data.radii, N * sizeof(float));
        cudaMalloc(&m_data.cellIDs, N * sizeof(float));
    }

    ~DeviceParticles() {
        cudaFree(m_data.positions);
        cudaFree(m_data.velocities);
        cudaFree(m_data.forces);

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
        cudaMemcpy(m_data.positions, hostParticles.positionsXYZ.data(),
                   m_data.N * 3 * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.velocities, hostParticles.velocitiesXYZ.data(),
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
        cudaMalloc(&m_data.cellIndex, cells * sizeof(int) + sizeof(int));
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

        cudaMemcpy(m_data.particlesPerCell, hostCells.particlesPerCell.data(),
                   hostCells.cellsTotal * sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(m_data.cellIndex, hostCells.cellIndex.data(),
                   hostCells.cellsTotal * sizeof(int) + sizeof(int), cudaMemcpyHostToDevice);

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
        m_data.maxForce = static_cast<float>(Tables.maxForce);
        m_data.stepSize = (m_data.r_max - m_data.r_min) / (m_data.tableElements - 1);

        cudaMalloc(&m_data.forceEnergyTable, 2 * m_data.tableElements * sizeof(float));
    }

    ~DeviceLookupTables() {
        cudaFree(m_data.forceEnergyTable);
    }

    DeviceLookupTables(const DeviceLookupTables&) = delete;
    DeviceLookupTables& operator=(const DeviceLookupTables&) = delete;

    LookupTableData data() const {
        return m_data;
    }

    void copyFromHost(SimTables& hostTables) const {
        /*Copies the force tables owned by a simulator to gpu for integration*/
        cudaMemcpy(m_data.forceEnergyTable, hostTables.forceEnergyTable.data(),
                   2 * m_data.tableElements * sizeof(float), cudaMemcpyHostToDevice);
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
    /*Helper for float3 operations*/
    return {arr1.x + arr2.x, arr1.y + arr2.y, arr1.z + arr2.z};
}


__device__ __forceinline__ float3 operator*(const float3& arr, const float scalar) {
    /*Helper for float3 operations*/
    return {arr.x * scalar, arr.y * scalar, arr.z * scalar};
}


// Put this in a device friendly format
__device__ __forceinline__ float3 getDistance(const float3& pos1, const float3& pos2, const float boxLength) {
    /*Takes two particles' positions in float3, and returns a float4 with the .x .y .z .w
    attributes equal to dx, dy, dz, and dr respectively. Handles periodic boundary conditions.*/
    float dx {pos1.x - pos2.x};
    dx -= boxLength * round(dx / boxLength);

    float dy {pos1.y - pos2.y};
    dy -= boxLength * round(dy / boxLength);

    float dz {pos1.z - pos2.z};
    dz -= boxLength * round(dz / boxLength);

    return {dx, dy, dz};
}


// Templates for now are unecessary, we'll see about the future though
template <typename T>
__device__ __forceinline__ float3 getVectorXYZ(const T* array, const int idx, const int N) {
    /*Returns a float3 of the specified idx from a flattened XYZ array, array has length 3N*/
    float3 vector;
    vector.x = array[idx];
    vector.y = array[idx + N];
    vector.z = array[idx + 2*N];

    return vector;
}


template <typename T>
__device__ __forceinline__ void writeVectorXYZ(const float3& vector, T* array, const int idx, const int N) {
    /*Writes a float3 of the specified idx to a flattened XYZ array, array has length 3N*/
    array[idx] = vector.x;
    array[idx + N] = vector.y;
    array[idx + 2*N] = vector.z;
}


template <typename T>
__device__ __forceinline__ void writeVectorXYZ(T* arr1, const int idx1, const int N1,
                                               T* arr2, const int idx2, const int N2) {
    /*Writes a vector from arr2 with idx2 and length 3N2 to arr1 with idx1 and length 3N1*/
    arr1[idx1] = arr2[idx2];
    arr1[idx1 + N1] = arr2[idx2 + N2];
    arr1[idx1 + 2*N1] = arr2[idx2 + 2*N2];
}


__device__ __forceinline__ int getCellID(float3 pos, int cellsPerSide, float boxLength) {
    // (0, 0, 0) is center of box but cellID 0 still starts in negative most corner
    // and increases first along +x, then +y, then +z
    // Periodic boundary condition is a half length around the box
    if (fabsf(pos.x) > (boxLength) || fabsf(pos.y) > (boxLength) || fabsf(pos.z) > (boxLength)) {
        return -1;
    }

    // Shift particle back to original box if in periodic boundary layer, boundary is half open for consistency
    if (pos.x >= boxLength / 2.0f) pos.x -= boxLength;
    if (pos.x < -boxLength / 2.0f) pos.x += boxLength;

    if (pos.y >= boxLength / 2.0f) pos.y -= boxLength;
    if (pos.y < -boxLength / 2.0f) pos.y += boxLength;

    if (pos.z >= boxLength / 2.0f) pos.z -= boxLength;
    if (pos.z < -boxLength / 2.0f) pos.z += boxLength;

    // Shift center coordinates to corner before calculating cell index
    float cellLength {boxLength / cellsPerSide};
    const int x_cell {static_cast<int>((pos.x + boxLength / 2.0f) / cellLength)};
    const int y_cell {static_cast<int>((pos.y + boxLength / 2.0f) / cellLength)};
    const int z_cell {static_cast<int>((pos.z + boxLength / 2.0f) / cellLength)};

    return x_cell + y_cell*cellsPerSide + z_cell*cellsPerSide*cellsPerSide;
}


__global__ void forceSum(ParticleData          Particles,
                         const CellData        Cells,
                         const LookupTableData LookupTables,
                         const DeviceParams    Params,
                         int*                  errorFlag) {
    /*Assume a maximum of 32 particles per cell (block) for now. Computes the sum of
    conservative forces for every particle by iterating over every particle in every
    neighbor cell.*/

    constexpr double MIN_CENTER_CENTER_DISTANCE{1e-3};
    constexpr int maxParticlesPerCell {32};
    __shared__ float sharedPos[3 * maxParticlesPerCell];

    // Bounds checking cellIDs
    const int cellsPerBlock {cuda::ceil_div(Params.cellsTotal, gridDim.x)};
    if ((blockIdx.x * cellsPerBlock >= Params.cellsTotal) || (*errorFlag))
        return;

    // First loop over each center cellID this block will process
    const int iterations {min(cellsPerBlock, Params.cellsTotal - (blockIdx.x * cellsPerBlock))}; // inline min()
    for (int cellID{blockIdx.x * cellsPerBlock}; cellID < (blockIdx.x * cellsPerBlock) + iterations; ++cellID) {
        
        __syncthreads();

        // Center cell case is special because we need to loop over every particle j != i
        // Particle i assigned to this thread and loading that position into shared mem
        int particlesPerCell {Cells.particlesPerCell[cellID]};
        const bool activeCenterThread {threadIdx.x < particlesPerCell};

        int particleID;
        float3 pos;
        float3 force{0, 0, 0};
        float potentialEnergy {0};

        if (activeCenterThread) {
            particleID = Cells.cellList[Cells.cellIndex[cellID] + threadIdx.x];
            pos = getVectorXYZ(Particles.positions, particleID, Particles.N);
            writeVectorXYZ(pos, sharedPos, threadIdx.x, particlesPerCell);
        }
        __syncthreads();

        if (activeCenterThread) {
            // Loop over every other particle in this cell
            for (int offset{1}; offset < particlesPerCell; ++offset) {
                const int idx {(offset + threadIdx.x) % particlesPerCell};
                const float3 otherPos {sharedPos[idx], sharedPos[idx + particlesPerCell], sharedPos[idx + 2*particlesPerCell]};
                const float3 distance {getDistance(pos, otherPos, Params.boxLength)};
                const float dr {distance.x*distance.x + distance.y*distance.y + distance.z*distance.z};

                if (dr < MIN_CENTER_CENTER_DISTANCE) {
                    atomicExch(errorFlag, 1);
                    return;
                }
                else if (dr < LookupTables.r_min) {
                    const float drInv {1.0f / dr};
                    force = force + distance * drInv * LookupTables.maxForce;
                    potentialEnergy += LookupTables.forceEnergyTable[LookupTables.tableElements]
                                    + LookupTables.maxForce * (LookupTables.r_min - dr);
                }
                else if (dr <= LookupTables.r_max) {
                    const float drInv {1.0f / dr};
                    const float forceMagnitude {LookupTables.forceEnergyTable[
                        static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))
                    ]};
                    force = force + distance * drInv * forceMagnitude;
                    potentialEnergy += LookupTables.forceEnergyTable[
                        static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize)
                        + LookupTables.tableElements)
                    ];
                }
            }
        }
        __syncthreads();

        // Then loop over each neighbor cell, +1 since we already did center cell
        for (int neighborIdx{cellID * 27 + 1}; neighborIdx < cellID * 27 + 27; ++neighborIdx) {

            const int neighborID {Cells.flatNeighborList[neighborIdx]};
            particlesPerCell = Cells.particlesPerCell[neighborID];

            // This thread loads a particle from neighbor cell into shared, this way previously inactive threads can do something useful
            if (threadIdx.x < particlesPerCell) {
                const int tempParticleID = Cells.cellList[Cells.cellIndex[neighborID] + threadIdx.x];
                writeVectorXYZ(sharedPos, threadIdx.x, particlesPerCell,
                               Particles.positions, tempParticleID, Particles.N);
            }
            __syncthreads();

            if (activeCenterThread) {
                // Loop over every particle in this neighbor cell
                for (int i{0}; i < particlesPerCell; ++i) {
                    const float3 otherPos {sharedPos[i], sharedPos[i + particlesPerCell], sharedPos[i + 2*particlesPerCell]};
                    const float3 distance {getDistance(pos, otherPos, Params.boxLength)};
                    const float dr {distance.x*distance.x + distance.y*distance.y + distance.z*distance.z};

                    if (dr < MIN_CENTER_CENTER_DISTANCE) {
                        atomicExch(errorFlag, 1);
                        return;
                    }
                    else if (dr < LookupTables.r_min) {
                        const float drInv {1.0f / dr};
                        force = force + distance * drInv * LookupTables.maxForce;
                        potentialEnergy += LookupTables.forceEnergyTable[LookupTables.tableElements]
                                        + LookupTables.maxForce * (LookupTables.r_min - dr);
                    }
                    else if (dr <= LookupTables.r_max) {
                        const float drInv {1.0f / dr};
                        const float forceMagnitude {LookupTables.forceEnergyTable[static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))]};
                        force = force + distance * drInv * forceMagnitude;
                        potentialEnergy += LookupTables.forceEnergyTable[
                            static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize)
                            + LookupTables.tableElements)
                        ];
                    }
                }
            }
            __syncthreads();
        }
        if (activeCenterThread) {
            writeVectorXYZ(force, Particles.forces, particleID, Particles.N);
        }
    }
}


template<class RNG> // Type of rng specified when calling kernel
__global__ void step1BAOA(ParticleData         Particles,
                         const CellData        Cells,
                         const LookupTableData LookupTables,
                         const DeviceParams    Params,
                         int*                  errorFlag,
                         RNG*                  rngState,
                         const unsigned long long seed) {
    /*Each thread assigned a particle, and does the half kick (B), both position drifts (A),
    and the random kick + drag (O) in the order BAOA. Then writes updated velocities and
    positions, and then rebuilds cell list. Also writes rng state for future steps*/
    const int particleID {blockDim.x * blockIdx.x + threadIdx.x};
    if ((particleID >= Particles.N) || (*errorFlag))
        return;

    float3 velocity {getVectorXYZ(Particles.velocities, particleID, Particles.N)};
    float3 position {getVectorXYZ(Particles.positions, particleID, Particles.N)};
    const float3 force {getVectorXYZ(Particles.forces, particleID, Particles.N)};

    velocity = velocity + force * (Params.dt / 2); // B, mass=1 in denominator
    position = position + velocity * (Params.dt / 2); // A

    const float c {expf(-Params.dt)}; // drag coef gamma=1 and mass=1, both in exponent
    const float sigma {sqrtf(5.0f * (1 - c)*(1 - c))}; // kT will likely non-dimensionalize

    curanddx::normal<float, curanddx::box_muller> normalDist(0, 1);
    RNG rng(seed, particleID, 0); // seed, subsequence, offset
    float4 rand4 {normalDist.generate4(rng)};
    float3 randomForce {rand4.x, rand4.y, rand4.z}; // Throwing away one value

    velocity = (velocity * c) + (randomForce * sigma); // O
    position = position + velocity * (Params.dt / 2); // A

    // Writing updated positions and velocities to global in preparation for force sum step
    writeVectorXYZ(position, Particles.positions, particleID, Particles.N);
    writeVectorXYZ(velocity, Particles.velocities, particleID, Particles.N);

    // Writing rng state for future time steps
    rngState[particleID] = rng;

    // Write new cellID to particle list and particlesPerCell
    const int newCellID {getCellID(position, Params.cellsPerSide, Params.boxLength)};
    if (newCellID == -1) { // If particle escapes box and PBC
        atomicExch(errorFlag, 1); // exchanges current value with 2nd argument
        return;
    }
    Particles.cellIDs[particleID] = newCellID;
    atomicAdd(Cells.particlesPerCell + newCellID, 1);
}


__global__ void buildCellList(const CellData     Cells,
                              const ParticleData Particles,
                              int*               errorFlag) {
    /*Writes particleID into compact cell list according to the recently updated cellIndex
    and cellIDs*/
    const int particleID {blockDim.x * blockIdx.x + threadIdx.x};
    if (particleID >= Particles.N || (*errorFlag)) {
        return;
    }
    const int cellID {Particles.cellIDs[particleID]};
    const int cellListIdx {atomicAdd(Cells.cellOffsets + cellID, 1)};

    Cells.cellList[cellListIdx] = particleID;
}


template<class RNG> // Type of rng specified when calling kernel
__global__ void stepBBAOA(ParticleData         Particles,
                         const CellData        Cells,
                         const LookupTableData LookupTables,
                         const DeviceParams    Params,
                         int*                  errorFlag,
                         RNG*                  rngState) {
    /*Each thread assigned a particle, and does the half kick (B), both position drifts (A),
    and the random kick + drag (O) in the order BAOA. Then writes updated velocities and
    positions, and then rebuilds cell list. Also writes rng state for future steps*/
    const int particleID {blockDim.x * blockIdx.x + threadIdx.x};
    if (particleID >= Particles.N || (*errorFlag))
        return;

    float3 velocity {getVectorXYZ(Particles.velocities, particleID, Particles.N)};
    float3 position {getVectorXYZ(Particles.positions, particleID, Particles.N)};
    const float3 force {getVectorXYZ(Particles.forces, particleID, Particles.N)};

    velocity = velocity + force * Params.dt; // BB, mass=1 in denominator
    position = position + velocity * (Params.dt / 2); // A

    const float c {expf(-Params.dt)}; // drag coef gamma=1 and mass=1, both in exponent
    const float sigma {sqrtf(5.0f * (1 - c)*(1 - c))}; // kT will likely non-dimensionalize, set to 5 rn

    curanddx::normal<float, curanddx::box_muller> normalDist(0, 1);
    RNG rng {rngState[particleID]};
    float4 rand4 {normalDist.generate4(rng)};
    float3 randomForce {rand4.x, rand4.y, rand4.z};

    velocity = (velocity * c) + (randomForce * sigma); // O
    position = position + velocity * (Params.dt / 2); // A

    // Writing updated positions and velocities to global in preparation for force sum step
    writeVectorXYZ(position, Particles.positions, particleID, Particles.N);
    writeVectorXYZ(velocity, Particles.velocities, particleID, Particles.N);

    // Writing rng state for future time steps
    rngState[particleID] = rng;

    // Write new cellID to particle list and particlesPerCell
    const int newCellID {getCellID(position, Params.cellsPerSide, Params.boxLength)};
    if (newCellID == -1) { // If particle escapes box and PBC
        atomicExch(errorFlag, 1); // exchanges current value with 2nd argument
        return;
    }
    Particles.cellIDs[particleID] = newCellID;
    atomicAdd(Cells.particlesPerCell + newCellID, 1);
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
    RNG* rngState {nullptr};
    cudaMalloc(&rngState, Sim->m_Params.N * sizeof(RNG));

    DeviceParticles Particles{Sim->m_Params.N};
    DeviceCells Cells{Sim->m_Params.N, Sim->m_Cells.cellsTotal};
    DeviceLookupTables Tables{Sim->m_Tables};
    DeviceParams Params{Sim};

    // For detecting if particle escapes box
    int* d_errorFlag {nullptr};
    cudaMalloc(&d_errorFlag, sizeof(int));
    cudaMemset(d_errorFlag, 0, sizeof(int));

    // Copying initial values from CPU to GPU
    Particles.copyFromHost(Sim->m_Particles);
    Cells.copyFromHost(Sim->m_Cells);
    Tables.copyFromHost(Sim->m_Tables);

    // Testing Kernel
    const int threads {32};
    const int blocks {cuda::ceil_div(Sim->m_Params.N, threads)};

    // Page-locked memory for positions for every frame write, always write frame 1
    float* framePositions {nullptr};
    cudaMallocHost(&framePositions, Sim->m_Params.N * 3 * sizeof(float));
    
    // Write frame 1, time step 0
    cudaMemcpy(framePositions, Particles.data().positions,
               Sim->m_Params.N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    writePositions(framePositions, Sim->m_Params.N, 1, true);

    // Do time step 1
    forceSum<<<blocks, threads>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag);

    // Need to reset particlesPerCell before rebuilding cellList
    cudaMemset(Cells.data().particlesPerCell, 0, Params.cellsTotal * sizeof(int));
    step1BAOA<<<blocks, threads>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag, rngState, seed);

    // Exclusive scan to generate cellIndex and writeOffsets, then kernel launch to write
    // particleID into compact cell list
    // Initial call to scan is to identify size of temporary scan storage, which is written into scanBytes
    size_t scanBytes {0};
    void* scanStorage {nullptr};
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  scanBytes,
                                  Cells.data().particlesPerCell,
                                  Cells.data().cellIndex,
                                  Params.cellsTotal);

    cudaMalloc(&scanStorage, scanBytes); // Allocate memory for temporary scan storage
    cub::DeviceScan::ExclusiveSum(scanStorage, // Actually scan particlesPerCell -> cellIndex
                                  scanBytes,
                                  Cells.data().particlesPerCell,
                                  Cells.data().cellIndex,
                                  Params.cellsTotal);

    // Initialize offsets used to write particleIDs to unique index in cellList
    cudaMemcpy(Cells.data().cellOffsets, Cells.data().cellIndex,
               Params.cellsTotal * sizeof(int), cudaMemcpyDeviceToDevice);

    // Launch kernel to write particleIDs into compact cellList
    buildCellList<<<blocks, threads>>>(Cells.data(), Particles.data(), d_errorFlag);

    // Number of time steps between each frame write, frames - 1 for initial write
    constexpr int ERROR_CHECK_INTERVAL = 50;
    const double frameSteps {static_cast<double>(Sim->m_Params.t_steps) / (Sim->m_Params.frames - 1)};
    for (int t{1}, frame{2}; t <= Sim->m_Params.t_steps; ++t) {
        std::cout << "Starting time step " << t << '\n';

        // Checking for particle out of bounds
        if (t % ERROR_CHECK_INTERVAL == 0) {
            int errorFlag;
            cudaMemcpy(&errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);

            if (errorFlag) {
                throw std::runtime_error("Particle escaped simulation box");
            }
        }

        // Write frame
        if (static_cast<int>(std::round(frameSteps * (frame - 1))) == t) {
            cudaMemcpy(framePositions, Particles.data().positions,
                       Sim->m_Params.N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            writePositions(framePositions, Sim->m_Params.N, frame, false);
            ++frame;
        }
        if (t == Sim->m_Params.t_steps)
            break;

        // Force sum step
        forceSum<<<blocks, threads>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag);

        // Need to reset particlesPerCell before rebuilding cellList
        cudaMemset(Cells.data().particlesPerCell, 0, Params.cellsTotal * sizeof(int));
        stepBBAOA<<<blocks, threads>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag, rngState);

        // Creat cellIndex, cellOffsets, and rebuild cellList
        cub::DeviceScan::ExclusiveSum(scanStorage,
                                      scanBytes,
                                      Cells.data().particlesPerCell,
                                      Cells.data().cellIndex,
                                      Params.cellsTotal);
        cudaMemcpy(Cells.data().cellOffsets, Cells.data().cellIndex,
                   Params.cellsTotal * sizeof(int), cudaMemcpyDeviceToDevice);
        buildCellList<<<blocks, threads>>>(Cells.data(), Particles.data(), d_errorFlag);
    }

    // Check for error flag if loop doesn't end on multiple of 50
    int errorFlag;
    cudaMemcpy(&errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if (errorFlag) {
        throw std::runtime_error("Particle escaped simulation box");
    }

    // Free rng state and scan storage, everything else frees automatically in destructor (RAII)
    cudaFree(rngState);
    cudaFree(scanStorage);
    cudaFreeHost(framePositions);
}
