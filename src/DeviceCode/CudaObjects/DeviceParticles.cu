#include "DeviceParticles.cuh"
#include "SimulatorDataStructs.hpp"


DeviceParticles::DeviceParticles(int N) {
    m_data.N = N;

    cudaMalloc(&m_data.positions, N * 3 * sizeof(float));
    cudaMalloc(&m_data.velocities, N * 3 * sizeof(float));
    cudaMalloc(&m_data.forces, N * 3 * sizeof(float));
    cudaMalloc(&m_data.potentialEnergies, N * sizeof(float));

    cudaMalloc(&m_data.radii, N * sizeof(float));
    cudaMalloc(&m_data.cellIDs, N * sizeof(float));
}


DeviceParticles::~DeviceParticles() {
    cudaFree(m_data.positions);
    cudaFree(m_data.velocities);
    cudaFree(m_data.forces);
    cudaFree(m_data.potentialEnergies);

    cudaFree(m_data.radii);
    cudaFree(m_data.cellIDs);
}


void DeviceParticles::copyFromHost(SimParticles& hostParticles) const {
    /*Copies the particle data owned by a simulator to gpu for integration*/
    cudaMemcpy(m_data.positions, hostParticles.positionsXYZ.data(),
                m_data.N * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(m_data.velocities, hostParticles.velocitiesXYZ.data(),
                m_data.N * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(m_data.potentialEnergies, hostParticles.potentialEnergies.data(),
                m_data.N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(m_data.radii, hostParticles.radii.data(),
                m_data.N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(m_data.cellIDs, hostParticles.cellIDs.data(),
                m_data.N * sizeof(int), cudaMemcpyHostToDevice);
}
