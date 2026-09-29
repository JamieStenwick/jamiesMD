#pragma once


struct SimParticles;


struct ParticleData {
    float* positions {nullptr};
    float* velocities {nullptr};
    float* forces {nullptr};
    float* potentialEnergies {nullptr};

    float* radii {nullptr};
    int* cellIDs {nullptr};

    int N;
};


class DeviceParticles {
    /*Unique owner of the gpu resources for a Particle Data struct
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    ParticleData m_data {};

public:
    explicit DeviceParticles(int N);
    ~DeviceParticles();

    DeviceParticles(const DeviceParticles&) = delete;
    DeviceParticles& operator=(const DeviceParticles&) = delete;

    ParticleData data() const {
        return m_data;
    }

    void copyFromHost(SimParticles& hostParticles) const;
};
