#pragma once
#include <vector>
#include "CudaKernels.cuh"

// Type aliases, if it gets long enough I'll move to it's own file
using PotentialFuncPtr = std::array<float, 2> (*)(float r, const std::vector<float>& args);

// non-Simulator function forward declarations, once it gets long enough I'll move to separate file
std::array<float, 2> getRminRmax (PotentialFuncPtr potential);
int getCellID(float x, float y, float z, int cellsPerSide, float boxLength);

struct SimParams {
    /*First 6 parameters are necessary for every simulation, remaining params
    are for the additional arguments to the potential function, must be passed in
    same order as unpacked in the potential function*/
    SimParams(int argc, char* args[]) :
        // Invariant arguments
        dt {std::strtof(args[1], nullptr)}, 
        vol_frac {std::strtof(args[2], nullptr)},
        temp {std::strtof(args[3], nullptr)},
        // long to int safe conversion on my computer
        N {static_cast<int>(strtol(args[4], nullptr, 10))},
        t_steps {static_cast<int>(std::strtol(args[5], nullptr, 10))},
        frames {static_cast<int>(std::strtol(args[6], nullptr, 10))},
        boxLength {static_cast<float>(std::pow(N * std::numbers::pi * (4.0/3.0) / vol_frac, 1.0/3.0))}

        {
            for (int i{7}; i < argc; ++i) {
                potentialArgs.push_back(std::strtof(args[i], nullptr));
            }
        }

    const float dt;
    const float vol_frac;
    const float temp;

    const int N;
    const int t_steps;
    const int frames;
    const float boxLength; // Derived from previous params

    std::vector<float> potentialArgs;
};


struct SimParticles {
    /*All data necessary to fully define particles. x, y, and z directions
    are laid out one after another (for now)*/
    SimParticles(const int N) :
        positionsXYZ(3*N),
        velocitiesXYZ(3*N),
        radii(N),
        cellIDs(N) // Derived from positions
        {}

    std::vector<float> positionsXYZ; // 3*N length, ALL x's first, ALL y's second, then ALL z's
    std::vector<float> velocitiesXYZ;
    std::vector<float> radii;
    std::vector<int> cellIDs;
};


struct SimCells {
    /*Container for cell list data*/
    SimCells(const int N, const float boxLength, const float r_max) :
        cellsPerSide {static_cast<int>(boxLength / r_max)}, // The static cast will truncate towards 0
        cellsTotal {cellsPerSide * cellsPerSide * cellsPerSide},
        cellList(N),
        cellIndex(cellsTotal + 1),
        flatNeighborList(cellsTotal * 27)
        {}
    const int cellsPerSide;
    const int cellsTotal;

    std::vector<int> cellList;
    std::vector<int> cellIndex;
    std::vector<int> flatNeighborList;
};


struct SimTables {
    /*Container for force and energy lookup tables*/
    SimTables(const int elements, PotentialFuncPtr potentialPtr) :
        N {elements},
        forceTable(N),
        energyTable(N),

        potential {potentialPtr},
        r_min {getRminRmax(potential)[0]},
        r_max {getRminRmax(potential)[1]}
        {}

    const int N;
    std::vector<float> forceTable;
    std::vector<float> energyTable;

    const PotentialFuncPtr potential {nullptr};
    const float r_min;
    const float r_max;
};


class Simulator {
public:
    Simulator(const SimParams& Params, const PotentialFuncPtr potential); // Using vector to hold all possible other args for now, will come back to this
    
    void populateLattice();
    void populateRandom(float offset);
    void writePositions() const;
    void integrate();

    friend void integrateKernelWrapper(Simulator* Sim);
    friend struct DeviceParams;

private:
    // All data has been grouped into structs
    const SimParams m_Params;
    SimTables m_Tables;
    SimParticles m_Particles;
    SimCells m_Cells;

    float getDistance(int particleID1, int particleID2) const; // Works
    void updateCellList(); // Works
    void fillLookupTables(); // Works
    void fillNeighborList();

};
