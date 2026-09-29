#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <array>
#include <numbers>


using PotentialFuncPtr = std::array<float, 2> (*)(float r, const std::vector<float>& args);
double getRmin (const std::vector<float>& args, const float thermalEnergy);
double getRmax (const float sigma);


struct SimParams {
    /*First 6 parameters are necessary for every simulation, remaining params
    are for the additional arguments to the potential function, must be passed in
    same order as unpacked in the potential function*/
    SimParams(int argc, char* args[]) :
        // Invariant arguments
        dt {std::strtof(args[1], nullptr)}, 
        vol_frac {std::strtof(args[2], nullptr)},
        kT {std::strtof(args[3], nullptr)},
        // long to int safe conversion on my computer
        N {static_cast<int>(strtol(args[4], nullptr, 10))},
        timeTotal {static_cast<int>(std::strtol(args[5], nullptr, 10))},
        frames {static_cast<int>(std::strtol(args[6], nullptr, 10))},
        timeSteps {static_cast<int>(timeTotal / dt)},
        boxLength {static_cast<float>(std::pow(N * std::numbers::pi * (4.0/3.0) / vol_frac, 1.0/3.0))}

        {
            for (int i{7}; i < argc; ++i) {
                potentialArgs.push_back(std::strtof(args[i], nullptr));
            }
        }

    const float dt;
    const float vol_frac;
    const float kT;

    const int N;
    const int timeTotal;
    const int frames;
    const int timeSteps; // Derived from dt and total time
    const float boxLength; // Derived from previous params

    std::vector<float> potentialArgs;
};


struct SimParticles {
    /*All data necessary to fully define particles. x, y, and z directions
    are laid out one after another (for now)*/
    SimParticles(const int N) :
        positionsXYZ(3*N),
        velocitiesXYZ(3*N),
        potentialEnergies(N),
        radii(N),
        cellIDs(N) // Derived from positions
        {}

    std::vector<float> positionsXYZ; // 3*N length, ALL x's first, ALL y's second, then ALL z's
    std::vector<float> velocitiesXYZ;
    std::vector<float> potentialEnergies;
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
        particlesPerCell(cellsTotal),
        flatNeighborList(cellsTotal * 27)
        {}
    const int cellsPerSide;
    const int cellsTotal;

    std::vector<int> cellList;
    std::vector<int> cellIndex;
    std::vector<int> particlesPerCell;
    std::vector<int> flatNeighborList;
};


struct SimTables {
    /*Container for force and energy lookup tables*/
    SimTables(const int elements, PotentialFuncPtr potentialPtr,
              const std::vector<float>& args, const float thermalEnergy) :
        N {elements},
        forceTable(N),
        energyTable(N),
        potential {potentialPtr},

        r_min {static_cast<float>(getRmin(args, thermalEnergy))},
        r_max {static_cast<float>(getRmax(args[1]))}
        {}

    const int N;
    std::vector<float> forceTable;
    std::vector<float> energyTable;
    const PotentialFuncPtr potential {nullptr};

    const float r_min;
    const float r_max;
};
