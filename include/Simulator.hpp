#pragma once


struct SimParams {
    /*Ordered doubles to reduce padding, constructor takes the vector of command line args
    and initializes the members with the values, MUST pass cmd-line args in same order as
    struct constructor. strtod and strtol require nullptr arg for some reason*/
    SimParams(char* args[]) :
        dt {std::strtod(args[1], nullptr)},
        vol_frac {std::strtod(args[2], nullptr)},
        temp {std::strtod(args[3], nullptr)},
        epsilon {std::strtod(args[4], nullptr)},
        sigma {std::strtod(args[5], nullptr)},

        // long to int safe conversion
        N {static_cast<int>(strtol(args[6], nullptr, 10))},
        t_steps {static_cast<int>(std::strtol(args[7], nullptr, 10))},
        frames {static_cast<int>(std::strtol(args[8], nullptr, 10))}
        {}

    const double dt {};
    const double vol_frac {};
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

    std::vector<double> positionsXYZ; // 3*N length x's first, y's second, then z's

    std::vector<double> velocitiesXYZ;

    std::vector<double> radii;
};


class Simulator {
public:
    Simulator(const SimParams& Params);
    
    void populateLattice();

    /*
    void populateRandom();
    */
   
    void writePositions() const;

private:

    const SimParams m_Params;
    const double m_boxLength; // Derived from m_Params

    Particles m_Particles;

    std::vector<double> m_cellList;
};
