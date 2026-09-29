#include "CudaWrappers.cuh"
#include "CudaKernels.cuh"
#include "CudaHelpers.cuh"
#include "SimUtilities.hpp"
#include "Simulator.hpp"
#include "FrameWrites.cuh"
#include <chrono>
#include <curanddx.hpp>
#include <cub/device/device_scan.cuh>
#include <iostream>


void rngKernelWrap(float* hostOut, const std::size_t N, const bool uniformFlag) {
    /*Takes a pointer to an array on the host CPU to copy results to, the size of the array,
    and the bounds and generates the requisite number of floats in a uniform distribution between
    0 and 1 if uniformFlag, and normally with mean 0 std 1 if not uniformFlag.*/
    // Allocate GPU memory
    float* devOut {nullptr};
    cudaMalloc(&devOut, N * sizeof(float));

    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() // type of generator, 4 at a time with this
                   + curanddx::PhiloxRounds<10>() // number of bijections
                   + curanddx::SM<JAMIESMD_CURANDDX_SM>() // specific architecture setting
                   + curanddx::Thread()); // initializes state on thread

    // Seed with system clock
    const unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Invoke kernel, 3 numbers per thread
    const std::size_t N_particles = N / 3;
    const int threads {256};
    const int blocks {static_cast<int>(cuda::ceil_div(N_particles, static_cast<std::size_t>(threads)))};
    rngKernel<RNG><<<blocks, threads>>>(devOut, seed, N, uniformFlag);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy results back to CPU
    cudaMemcpy(hostOut, devOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(devOut);
}


void errorFlagCheck(int* d_errorFlag) {
    /*Takes the pointer to device error flag and throws the appropriate exception if
    an error has been flagged.*/
    int errorFlag;
    cudaMemcpy(&errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);

    switch (errorFlag) {
    case 1:
        throw std::runtime_error("Illegal particle overlap, try reducing dt");
        break;
    case 2:
        throw std::runtime_error("Too many particles in a cell, try reducing phi");
        break;
    case 3:
        throw std::runtime_error("Particle escaped simulation box, try lowering potential");
        break;
    }
}


void rebuildCellList(void* scanStorage,
                     size_t scanBytes,
                     DeviceParticles& Particles,
                     DeviceCells& Cells,
                     const int cellsTotal,
                     const int blocks,
                     const int threads,
                     int* d_errorFlag) {
    /*Wrapper for scanning + rebuilding cell list*/
    cub::DeviceScan::ExclusiveSum(scanStorage,
                                  scanBytes,
                                  Cells.data().particlesPerCell,
                                  Cells.data().cellIndex,
                                  cellsTotal);

    // Initialize offsets used to write particleIDs to unique index in cellList
    cudaMemcpy(Cells.data().cellOffsets, Cells.data().cellIndex,
               cellsTotal * sizeof(int), cudaMemcpyDeviceToDevice);

    // Launch kernel to write particleIDs into compact cellList
    buildCellList<<<blocks, threads>>>(Cells.data(), Particles.data(), d_errorFlag);
}


int getMaxBlocks(ForceKernelPtr, const int threads, const size_t dynamicSharedMem) {
    /*Calculates maximum concurrent blocks from gpu properties and kernel information.*/
    int blocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, forceSum, threads, dynamicSharedMem);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    return blocksPerSM * prop.multiProcessorCount;
}


void integrateKernelWrapper(Simulator* Sim, const bool writeEnergy) {
    /*Takes a pointer to a Simulator object, allocates device memory for integration,
    and orchestrates Simulator host actions and Integrator Device actions during the
    duration of the integration. Right now integrates from start to finish.*/

    // Random number generation setup
    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>()
                   + curanddx::PhiloxRounds<6>()
                   + curanddx::SM<JAMIESMD_CURANDDX_SM>()
                   + curanddx::Thread());
    const unsigned long long seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Allocating GPU memory
    RNG* rngState {nullptr};
    int* d_errorFlag {nullptr}; // One flag for multiple errors

    cudaMalloc(&rngState, Sim->m_Params.N * sizeof(RNG));
    cudaMalloc(&d_errorFlag, sizeof(int));

    DeviceParticles Particles{Sim->m_Params.N};
    DeviceCells Cells{Sim->m_Params.N, Sim->m_Cells.cellsTotal};
    DeviceLookupTables Tables{Sim->m_Tables};
    DeviceParams Params{Sim};
    FrameWrites Frames {Sim->m_Params.N};

    // Copying initial values from CPU to GPU and setting errorFlag
    Particles.copyFromHost(Sim->m_Particles);
    Cells.copyFromHost(Sim->m_Cells);
    Tables.copyFromHost(Sim->m_Tables);
    cudaMemset(d_errorFlag, 0, sizeof(int));
    
    // Kernel parameters, maxConcurrentBlocks is for force kernel only
    constexpr int THREADS {64};
    const size_t sharedPosBytes {3 * THREADS * sizeof(float)};
    const int particleBlocks {cuda::ceil_div(Sim->m_Params.N, THREADS)};
    const int maxConcurrentBlocks {getMaxBlocks(forceSum, THREADS, sharedPosBytes)};

    // Write frame 1, time step 0
    forceSum<<<maxConcurrentBlocks, THREADS, sharedPosBytes>>>(Particles.data(), Cells.data(), Tables.data(),
                                                               Params, d_errorFlag, writeEnergy);

    Frames.copyFromDevice(Particles, writeEnergy);
    writePositions(Frames.getPosition(), Sim->m_Params.N, true);
    if (writeEnergy)
        writeEnergies(Frames.getEnergy(), Frames.getVelocity(), Frames.getForce(), Sim->m_Params.N, Sim->m_Params.dt, true);

    // Do time step 1, Need to reset particlesPerCell before rebuilding cellList
    cudaMemset(Cells.data().particlesPerCell, 0, Params.cellsTotal * sizeof(int));
    step1BAOA<<<particleBlocks, THREADS>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag, rngState, seed);

    // Initial call to scan is to identify size of temporary scan storage, which is written into scanBytes
    size_t scanBytes {0};
    void* scanStorage {nullptr};
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  scanBytes,
                                  Cells.data().particlesPerCell,
                                  Cells.data().cellIndex,
                                  Params.cellsTotal);
    cudaMalloc(&scanStorage, scanBytes); // Allocate memory for temporary scan storage
    rebuildCellList(scanStorage, scanBytes, Particles, Cells,
                    Params.cellsTotal, particleBlocks, THREADS, d_errorFlag);

    // Number of time steps between each frame write, frames - 1 for initial write
    constexpr int ERROR_CHECK_INTERVAL = 5000;
    const double frameSteps {static_cast<double>(Sim->m_Params.timeSteps) / (Sim->m_Params.frames - 1)};
    for (int t{1}, frame{2}; t <= Sim->m_Params.timeSteps; ++t) {
        if (t % ERROR_CHECK_INTERVAL == 0)
            errorFlagCheck(d_errorFlag);
        
        const bool writeStep(static_cast<int>(std::round(frameSteps * (frame - 1))) == t);
        const bool energyFrame(writeEnergy && writeStep);
        forceSum<<<maxConcurrentBlocks, THREADS, sharedPosBytes>>>(Particles.data(), Cells.data(), Tables.data(),
                                                                   Params, d_errorFlag, energyFrame);

        // Write frame
        if (writeStep) {
            std::cout << "Writing frame " << frame << '/' << Sim->m_Params.frames << '\n';
            Frames.copyFromDevice(Particles, writeEnergy);
            writePositions(Frames.getPosition(), Sim->m_Params.N, false);
            if (writeEnergy)
                writeEnergies(Frames.getEnergy(), Frames.getVelocity(), Frames.getForce(), Sim->m_Params.N, Sim->m_Params.dt, false);
            ++frame;
        }

        // This part is unnecessary for the last time step
        cudaMemset(Cells.data().particlesPerCell, 0, Params.cellsTotal * sizeof(int));
        stepBBAOA<<<particleBlocks, THREADS>>>(Particles.data(), Cells.data(), Tables.data(), Params, d_errorFlag, rngState);
        rebuildCellList(scanStorage, scanBytes, Particles, Cells,
                    Params.cellsTotal, particleBlocks, THREADS, d_errorFlag);
    }

    // Check for error flag if loop doesn't end on multiple of 50
    errorFlagCheck(d_errorFlag);

    // Free rng state and scan storage, everything else frees automatically in destructor (RAII)
    cudaFree(rngState);
    cudaFree(scanStorage);
    cudaFree(d_errorFlag);
}
