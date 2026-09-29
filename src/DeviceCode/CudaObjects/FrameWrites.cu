#include "FrameWrites.cuh"


FrameWrites::FrameWrites(int N) :
    N {N}
{
    cudaMallocHost(&positions, N * 3 * sizeof(float));
    cudaMallocHost(&velocities, N * 3 * sizeof(float));
    cudaMallocHost(&forces, N * 3 * sizeof(float));
    cudaMallocHost(&potentialEnergies, N * sizeof(float));
}


FrameWrites::~FrameWrites() {
    cudaFreeHost(positions);
    cudaFreeHost(velocities);
    cudaFreeHost(forces);
    cudaFreeHost(potentialEnergies);
}


void FrameWrites::copyFromDevice(DeviceParticles& Particles, const bool writeEnergy) const {
    /*Copies the position velocity and potential energy data from the gpu to the cpu*/
    cudaMemcpy(positions, Particles.data().positions,
                N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    if (writeEnergy) {
        cudaMemcpy(velocities, Particles.data().velocities,
                    N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(forces, Particles.data().forces,
                    N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(potentialEnergies, Particles.data().potentialEnergies,
                    N * sizeof(float), cudaMemcpyDeviceToHost);
    }
}
