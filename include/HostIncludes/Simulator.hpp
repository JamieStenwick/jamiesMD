#pragma once
#include "SimulatorDataStructs.hpp"
#include <vector>

// Type aliases, if it gets long enough I'll move to it's own file
using PotentialFuncPtr = std::array<float, 2> (*)(float r, const std::vector<float>& args);


class Simulator {
public:
    Simulator(const SimParams& Params, const PotentialFuncPtr potential);
    
    void populateLattice();
    void populateRandom(float offset);
    void integrate(const bool writeEnergy);

    friend void integrateKernelWrapper(Simulator* Sim, const bool writeEnergy);
    friend struct DeviceParams;

private:
    // All data has been grouped into structs
    const SimParams m_Params;
    SimTables m_Tables;
    SimParticles m_Particles;
    SimCells m_Cells;

    float getDistance(int particleID1, int particleID2) const; // Works, make a pure function later
    void updateCellList(); // Works
    void fillLookupTables(); // Works
    void fillNeighborList(); // Works
    void zeroNetMomentum();
};
