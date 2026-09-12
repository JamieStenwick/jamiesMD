#pragma once
#include <vector>

// Type aliases, if it gets long enough I'll move to it's own file
using PotentialFuncPtr = std::array<float, 2> (*)(float r, std::vector<float> args);

// non-Simulator function forward declarations, once it gets long enough I'll move to separate file
std::array<float, 2> getRminRmax (PotentialFuncPtr potential);
int getCellID(float x, float y, float z, float r_min, float boxLength);
std::vector<int> getNeighborList(const int cellID, const int cellsPerSide);

struct SimParams {
    /*Ordered doubles to reduce padding, constructor takes the vector of command line args
    and initializes the members with the values, MUST pass cmd-line args in same order as
    struct constructor. strtod and strtol require nullptr arg for some reason*/
    SimParams(char* args[]) :
        dt {std::strtod(args[1], nullptr)},
        vol_frac {std::strtof(args[2], nullptr)},
        temp {std::strtod(args[3], nullptr)},
        epsilon {std::strtod(args[4], nullptr)},
        sigma {std::strtod(args[5], nullptr)},

        // long to int safe conversion
        N {static_cast<int>(strtol(args[6], nullptr, 10))},
        t_steps {static_cast<int>(std::strtol(args[7], nullptr, 10))},
        frames {static_cast<int>(std::strtol(args[8], nullptr, 10))}
        {}

    const double dt {};
    const float vol_frac {};
    const double temp {};
    const double epsilon {};
    const double sigma {};
    const int N {};
    const int t_steps {};
    const int frames {};
};


struct Particles {
    /*All data necessary to fully define particles. x, y, and z directions
    are laid out one after another (for now)*/
    Particles(const int N) :
        positionsXYZ(3*N),
        velocitiesXYZ(3*N),
        radii(N)
        {}

    std::vector<float> positionsXYZ; // 3*N length x's first, y's second, then z's
    std::vector<float> velocitiesXYZ;
    std::vector<float> radii;
};


class Simulator {
public:
    Simulator(const SimParams& Params, const PotentialFuncPtr potential); // Using vector to hold all possible other args for now, will come back to this
    
    void populateLattice();

    void populateRandom(float offset);
   
    void writePositions() const;

private:

    const SimParams m_Params;
    const float m_boxLength; // Derived from m_Params
    const PotentialFuncPtr m_potential {nullptr};

    Particles m_Particles;
    std::vector<float> m_cellList;

    float getDistance(int particleID1, int particleID2) const;

};
