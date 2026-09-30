#pragma once
#include "CudaHelpers.cuh"
#include <curanddx.hpp>
#include "CudaObjects.cuh"
#include <cuda/cmath>


template<class RNG>
__global__ void rngKernel(float* devOut,
                          const unsigned long long seed,
                          const std::size_t N,
                          const bool uniformFlag) {
    /*Takes pointer to GPU output array, the seed, length of the output, and the uniform
    flag as input and writes an array of random numbers between 0 and 1 uniformly if
    uniform and with mean 0 std 1 if normal.*/
    const std::size_t i {blockDim.x * blockIdx.x + threadIdx.x};
    const std::size_t j {3 * i};

    // Bounds checking make sure our indices are valid
    if (j >= N)
        return;
    
    // Subsequence and offset are ways to index into the RNG sequence.
    // Both subsequence and offset are 64-bit, so really big
    RNG rng(seed, i, 0);
    float4 rand;
    if (uniformFlag) {
        curanddx::uniform<float> uniformDist(0, 1);
        rand = uniformDist.generate4(rng);
    }
    else {
        curanddx::normal<float, curanddx::box_muller> normalDist(0, 1);
        rand = normalDist.generate4(rng);
    }

    devOut[j] = rand.x;
    devOut[j + 1] = rand.y;
    devOut[j + 2] = rand.z;
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
    const std::size_t index {static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x};
    if ((index >= static_cast<std::size_t>(Particles.N)) || (*errorFlag))
        return;
    const int particleID {static_cast<int>(index)}; // Bounds check guarantees this fits in int.

    float3 velocity {getVectorXYZ(Particles.velocities, particleID, Particles.N)};
    float3 position {getVectorXYZ(Particles.positions, particleID, Particles.N)};
    const float3 force {getVectorXYZ(Particles.forces, particleID, Particles.N)};

    velocity = velocity + force * (Params.dt / 2); // B, mass=1 in denominator
    position = position + velocity * (Params.dt / 2); // A

    curanddx::normal<float, curanddx::box_muller> normalDist(0, 1);
    RNG rng(seed, particleID, 0); // seed, subsequence, offset
    float4 rand4 {normalDist.generate4(rng)};
    float3 randomForce {rand4.x, rand4.y, rand4.z}; // Throwing away one value

    velocity = (velocity * Params.c) + (randomForce * Params.sigma); // O
    position = position + velocity * (Params.dt / 2); // A

    // Writing updated positions and velocities to global in preparation for force sum step
    writeVectorXYZ(position, Particles.positions, particleID, Particles.N);
    writeVectorXYZ(velocity, Particles.velocities, particleID, Particles.N);

    // Writing rng state for future time steps
    rngState[particleID] = rng;

    // Write new cellID to particle list and particlesPerCell
    const int newCellID {getCellID(position, Params.cellsPerSide, Params.boxLength)};
    if (newCellID == -1) { // If particle escapes box and PBC
        atomicExch(errorFlag, 3); // exchanges current value with 2nd argument
        return;
    }
    Particles.cellIDs[particleID] = newCellID;
    atomicAdd(Cells.particlesPerCell + newCellID, 1);
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
    const std::size_t index {static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x};
    if (index >= static_cast<std::size_t>(Particles.N) || (*errorFlag))
        return;
    const int particleID {static_cast<int>(index)}; // Bounds check guarantees this fits in int.

    float3 velocity {getVectorXYZ(Particles.velocities, particleID, Particles.N)};
    float3 position {getVectorXYZ(Particles.positions, particleID, Particles.N)};
    const float3 force {getVectorXYZ(Particles.forces, particleID, Particles.N)};

    velocity = velocity + force * Params.dt; // BB, mass=1 in denominator
    position = position + velocity * (Params.dt / 2); // A

    curanddx::normal<float, curanddx::box_muller> normalDist(0, 1);
    RNG rng {rngState[particleID]};
    float4 rand4 {normalDist.generate4(rng)};
    float3 randomForce {rand4.x, rand4.y, rand4.z};

    velocity = (velocity * Params.c) + (randomForce * Params.sigma); // O
    position = position + velocity * (Params.dt / 2); // A

    // Writing updated positions and velocities to global in preparation for force sum step
    writeVectorXYZ(position, Particles.positions, particleID, Particles.N);
    writeVectorXYZ(velocity, Particles.velocities, particleID, Particles.N);

    // Writing rng state for future time steps
    rngState[particleID] = rng;

    // Write new cellID to particle list and particlesPerCell
    const int newCellID {getCellID(position, Params.cellsPerSide, Params.boxLength)};
    if (newCellID == -1) {
        atomicExch(errorFlag, 3); // exchanges current value with 2nd argument
        return;
    }
    Particles.cellIDs[particleID] = newCellID;
    atomicAdd(Cells.particlesPerCell + newCellID, 1);
}


__global__ void forceSum(ParticleData          Particles,
                         const CellData        Cells,
                         const LookupTableData LookupTables,
                         const DeviceParams    Params,
                         int*                  errorFlag,
                         const bool            writeEnergy) {
    /*Assume a maximum of 32 particles per cell (block) for now. Computes the sum of
    conservative forces for every particle by iterating over every particle in every
    neighbor cell.*/

    constexpr double MIN_CENTER_CENTER_DISTANCE{1e-3};
    extern __shared__ float sharedPos[];

    // Bounds checking cellIDs
    // The ceiling quotient cannot exceed the int-sized cell count.
    const int cellsPerBlock {static_cast<int>(cuda::ceil_div(Params.cellsTotal, gridDim.x))};
    const std::size_t firstCell {static_cast<std::size_t>(blockIdx.x) * cellsPerBlock};
    if ((firstCell >= static_cast<std::size_t>(Params.cellsTotal)) || (*errorFlag))
        return;
    const int startCell {static_cast<int>(firstCell)};

    // First loop over each center cellID this block will process
    const int iterations {min(cellsPerBlock, Params.cellsTotal - startCell)}; // inline min()
    for (int cellID{startCell}; cellID < startCell + iterations; ++cellID) {
        
        __syncthreads();

        // Center cell case is special because we need to loop over every particle j != i
        // Particle i assigned to this thread and loading that position into shared mem
        int particlesPerCell {Cells.particlesPerCell[cellID]};
        if (particlesPerCell > blockDim.x) {
            atomicExch(errorFlag, 2);
            return;
        }
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
                const int idx {(offset + static_cast<int>(threadIdx.x)) % particlesPerCell};
                const float3 otherPos {sharedPos[idx], sharedPos[idx + particlesPerCell], sharedPos[idx + 2*particlesPerCell]};
                const float3 distance {getDistance(pos, otherPos, Params.boxLength)};
                const float dr {sqrtf(distance.x*distance.x + distance.y*distance.y + distance.z*distance.z)};

                if (dr < MIN_CENTER_CENTER_DISTANCE) {
                    atomicExch(errorFlag, 1);
                    return;
                }
                else if (dr < LookupTables.r_min) {
                    const float drInv {1.0f / dr};
                    force = force + distance * drInv * LookupTables.maxForce;
                    if (writeEnergy) {
                        potentialEnergy += LookupTables.energyTable[0]
                                        + LookupTables.maxForce * (LookupTables.r_min - dr);
                    }
                }
                else if (dr <= LookupTables.r_max) {
                    const float drInv {1.0f / dr};
                    const float forceMagnitude {LookupTables.forceTable[
                        static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))
                    ]};
                    force = force + distance * drInv * forceMagnitude;
                    if (writeEnergy) {
                        potentialEnergy += LookupTables.energyTable[
                            static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))
                        ];
                    }
                }
            }
        }
        __syncthreads();

        // Then loop over each neighbor cell, +1 since we already did center cell
        for (int neighborIdx{cellID * 27 + 1}; neighborIdx < cellID * 27 + 27; ++neighborIdx) {

            const int neighborID {Cells.flatNeighborList[neighborIdx]};
            particlesPerCell = Cells.particlesPerCell[neighborID];
            if (particlesPerCell > blockDim.x) {
                atomicExch(errorFlag, 2);
                return;
            }

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
                    const float dr {sqrtf(distance.x*distance.x + distance.y*distance.y + distance.z*distance.z)};

                    if (dr < MIN_CENTER_CENTER_DISTANCE) {
                        atomicExch(errorFlag, 1);
                        return;
                    }
                    else if (dr < LookupTables.r_min) {
                        const float drInv {1.0f / dr};
                        force = force + distance * drInv * LookupTables.maxForce;
                        if (writeEnergy) {
                            potentialEnergy += LookupTables.energyTable[0]
                                            + LookupTables.maxForce * (LookupTables.r_min - dr);
                        }
                    }
                    else if (dr <= LookupTables.r_max) {
                        const float drInv {1.0f / dr};
                        const float forceMagnitude {LookupTables.forceTable[static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))]};
                        force = force + distance * drInv * forceMagnitude;
                        if (writeEnergy) {
                            potentialEnergy += LookupTables.energyTable[
                                static_cast<size_t>(round((dr - LookupTables.r_min) / LookupTables.stepSize))
                            ];
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (activeCenterThread) {
            writeVectorXYZ(force, Particles.forces, particleID, Particles.N);
            if (writeEnergy)
                Particles.potentialEnergies[particleID] = potentialEnergy;
        }
    }
}


__global__ void buildCellList(const CellData     Cells,
                              const ParticleData Particles,
                              int*               errorFlag) {
    /*Writes particleID into compact cell list according to the recently updated cellIndex
    and cellIDs*/
    const std::size_t index {static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x};
    if (index >= static_cast<std::size_t>(Particles.N) || (*errorFlag)) {
        return;
    }
    const int particleID {static_cast<int>(index)}; // Bounds check guarantees this fits in int.
    const int cellID {Particles.cellIDs[particleID]};
    const int cellListIdx {atomicAdd(Cells.cellOffsets + cellID, 1)};

    Cells.cellList[cellListIdx] = particleID;
}
