#include "DeviceCells.cuh"
#include "SimulatorDataStructs.hpp"


DeviceCells::DeviceCells(int N, int cells) {
    cudaMalloc(&m_data.cellList, N * sizeof(int));
    cudaMalloc(&m_data.particlesPerCell, cells * sizeof(int));
    cudaMalloc(&m_data.cellIndex, cells * sizeof(int) + sizeof(int));
    cudaMalloc(&m_data.cellOffsets, cells * sizeof(int));
    cudaMalloc(&m_data.flatNeighborList, cells * 27 * sizeof(int));
}

DeviceCells::~DeviceCells() {
    cudaFree(m_data.cellList);
    cudaFree(m_data.particlesPerCell);
    cudaFree(m_data.cellIndex);
    cudaFree(m_data.cellOffsets);
    cudaFree(m_data.flatNeighborList);
}

// Change to just a reference to SimCell struct when you make it
void DeviceCells::copyFromHost(SimCells& hostCells) const {
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
