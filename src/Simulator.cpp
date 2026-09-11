#include <vector>
#include <cstdlib>
#include <numbers>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <exception>
#include <utility>
#include <array>
#include "Simulator.hpp"

Simulator::Simulator(const SimParams& Params) :
        m_Params {Params},
        m_boxLength {std::pow(m_Params.N * std::numbers::pi * (4.0/3.0) / m_Params.vol_frac, 1.0/3.0)},

        m_Particles {m_Params.N},
        m_cellList(m_Params.N)
        {}

void Simulator::populateLattice() {
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
void populateRandom() {
    1. Get exactly number of random velocities (3*N) probably normal distribution and scaled by temp and fill velocity arrays.
    2. Then get enough uniform numbers for twice number of particles (3*N*2)
    3. Loop through each particle on CPU (GPU just a means of generating a shitload of random numbers rn)
    3a. Look through every particle in cell corresponding to generated number and see if sampled position < allowable delta r
    3b. If valid, append particle to cell list and position to particle, if not go back to 3
    4. Repeat until all particles assigned valid location
    5. If random position list is exhausted, relaunch kernel and copy back just an N number of positions (3*N) total
    
}
*/

void Simulator::writePositions() const {
    // Remember to try{} this, also will open in binary mode for during runtime
    static std::ofstream outfile{"positions.txt"};
    // if (!outfile) {throw std::runtime_error("outfile could not be opened");}

    for (int i{0}; i < m_Params.N; i++) {
        outfile << m_xpos[i] << ' ' << m_ypos[i] << ' ' << m_zpos[i] << '\n';
    }
    outfile.close();
}
