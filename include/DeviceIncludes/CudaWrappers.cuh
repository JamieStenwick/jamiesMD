#pragma once
#include "CudaObjects.cuh"


using ForceKernelPtr = void (*)(ParticleData Particles,
                                const CellData Cells,
                                const LookupTableData LookupTables,
                                const DeviceParams Params,
                                int* errorFlag,
                                const bool writeEnergy);

class Simulator;

void rngKernelWrap(float* output, const std::size_t N, const bool uniformFlag);

// Helpers for the integrate wrapper
void errorFlagCheck(int* d_errorFlag);
void rebuildCellList(void* scanStorage,
                     size_t scanBytes,
                     DeviceParticles& Particles,
                     DeviceCells& Cells,
                     const int cellsTotal,
                     const int blocks,
                     const int threads,
                     int* d_errorFlag);
int getMaxBlocks(ForceKernelPtr, const int threads, const size_t dynamicSharedMem);

void integrateKernelWrapper(Simulator* Sim, const bool writeEnergy);
