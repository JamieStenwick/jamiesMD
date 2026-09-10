#include <vector>
#include <cstdlib>
#include <numbers>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <exception>
#include <utility>
#include <array>


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
    double dt {};
    double vol_frac {};
    double temp {};
    double epsilon {};
    double sigma {};
    int N {};
    int t_steps {};
    int frames {};
};


struct Particles {
    /*All data necessary to fully define particles, 3 coordinate position arrays,
    3 coordinate velocity arrays, and a radii array (for now)*/
    std::vector<double> x{};
    std::vector<double> y{};
    std::vector<double> z{};

    std::vector<double> vx{};
    std::vector<double> vy{};
    std::vector<double> vz{};

    std::vector<double> r{};
};


class Simulator {
public:
    Simulator(const SimParams& Params) :
        m_Params {Params},
        m_boxLength {std::pow(m_Params.N * std::numbers::pi * (4.0/3.0) / m_Params.vol_frac, 1.0/3.0)},

        m_xpos(m_Params.N),
        m_ypos(m_Params.N),
        m_zpos(m_Params.N),

        m_radii(m_Params.N, 1),
        m_cellList(m_Params.N)
        {}
    
    void setPositionsLattice() {
        /*Orders the particles mostly evenly spaced on a cubic lattice.
        The spacing on the edges being 1/2 such that when accounting for
        the 'ghost image' from the other side it will match inner spacing.*/
        const int root {static_cast<int>(std::cbrt(m_Params.N))}; 
        std::pair<int, int> particlesYZ {root, root};

        if ((root + 1)*particlesYZ.first*particlesYZ.second < m_Params.N)
            ++particlesYZ.first;
        if ((root + 1)*particlesYZ.first*particlesYZ.second < m_Params.N)
            ++particlesYZ.second;
        
        // Particles are evenly spaced along Y and Z axes, and spacing on x axis is either L/(root) or L/(root+1)
        std::pair<double, double> particleSpacing {m_boxLength/particlesYZ.first, m_boxLength/particlesYZ.second};
        std::array<int, 3> pos {0, 0, 0};

        int extras {m_Params.N - root*particlesYZ.first*particlesYZ.second};
        int i{0};
        // First we space according to (root + 1) particles per X row
        while (extras > 0) {
            // Whenever extras runs out, the first particle in the row will
            // not be evenly spaced with the rest, this is inconsequential.
            if (pos[0] == root + 1) {
                pos[0] = 0;
                ++pos[1];
                --extras;
            }
            if (pos[1] == particlesYZ.first) {
                pos[1] = 0;
                ++pos[2];
            }

            m_xpos[i] = pos[0] * m_boxLength/(root + 1) + (m_boxLength/(root + 1) / 2);
            m_ypos[i] = pos[1] * particleSpacing.first + (particleSpacing.first / 2);
            m_zpos[i] = pos[2] * particleSpacing.second + (particleSpacing.second / 2);

            ++i, ++pos[0];
        }

        // Once we run out of extras to fill, fill the rest of the X rows root # of particles
        while (i < m_Params.N) {
            if (pos[0] == root) {
                pos[0] = 0;
                ++pos[1];
            }
            if (pos[1] == particlesYZ.first) {
                pos[1] = 0;
                ++pos[2];
            }

            m_xpos[i] = pos[0] * m_boxLength / root + (m_boxLength / root / 2);
            m_ypos[i] = pos[1] * particleSpacing.first + (particleSpacing.first / 2);
            m_zpos[i] = pos[2] * particleSpacing.second + (particleSpacing.second / 2);

            ++i, ++pos[0];
        }
    }

    /*
    void setPositionsRandom() {
        1. Get exactly number of random velocities (3*N) probably normal distribution and scaled by temp and fill velocity arrays.
        2. Then get enough uniform numbers for twice number of particles (3*N*2)
        3. Loop through each particle on CPU (GPU just a means of generating a shitload of random numbers rn)
        3a. Look through every particle in cell corresponding to generated number and see if sampled position < allowable delta r
        3b. If valid, append particle to cell list and position to particle, if not go back to 3
        4. Repeat until all particles assigned valid location
        5. If random position list is exhausted, relaunch kernel and copy back just an N number of positions (3*N) total
        
    }
    */
   
    void writePositions() const {
        // Remember to try{} this, also will open in binary mode for during runtime
        static std::ofstream outfile{"positions.txt"};
        // if (!outfile) {throw std::runtime_error("outfile could not be opened");}

        for (int i{0}; i < m_Params.N; i++) {
            outfile << m_xpos[i] << ' ' << m_ypos[i] << ' ' << m_zpos[i] << '\n';
        }
        outfile.close();
    }

private:

    const SimParams m_Params;
    const double m_boxLength{};

    std::vector<double> m_xpos{};
    std::vector<double> m_ypos{};
    std::vector<double> m_zpos{};

    std::vector<double> m_radii{};
    std::vector<double> m_cellList{};
};
