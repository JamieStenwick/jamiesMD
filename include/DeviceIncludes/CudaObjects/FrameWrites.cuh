#pragma once
#include "DeviceParticles.cuh"


class FrameWrites {
    /*Manages the page-locked memory for writing positions and/or energies for each frame*/
private:
    float* positions {nullptr};
    float* velocities {nullptr};
    float* forces {nullptr};
    float* potentialEnergies {nullptr};

    int N;

public:
    explicit FrameWrites(int N);
    ~FrameWrites();

    FrameWrites(const FrameWrites&) = delete;
    FrameWrites& operator=(const FrameWrites&) = delete;

    void copyFromDevice(DeviceParticles& Particles, const bool writeEnergy) const;

    float* getPosition() {return positions;}
    float* getVelocity() {return velocities;}
    float* getForce() {return forces;}
    float* getEnergy() {return potentialEnergies;}
};
