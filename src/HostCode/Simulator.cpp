#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include "Simulator.hpp"
#include "SimUtilities.hpp"
#include "CudaWrappers.cuh"
#include <iostream>


constexpr int N_TABLE_ELEMENTS {50000};


Simulator::Simulator(const SimParams& Params, const PotentialFuncPtr potential) : // Using vector to hold all possible other args for now, will come back to this
        m_Params {Params},
        m_Tables {N_TABLE_ELEMENTS, potential, Params.potentialArgs, Params.kT},
        m_Particles {m_Params.N},
        m_Cells {m_Params.N, m_Params.boxLength, m_Tables.r_max}
        {
            // Filling energy and force tables
            fillLookupTables();

            // Filling flat neighbor list
            fillNeighborList();
        }


void Simulator::populateLattice() {
    /*Orders the particles mostly evenly spaced on a cubic lattice.
    The spacing on the edges being 1/2 such that when accounting for
    the 'ghost image' from the other side it will match inner spacing.
    Also updates the cellID list in Particless struct*/
    std::vector<float>& posXYZ {m_Particles.positionsXYZ}; // shorter name
    const int root {static_cast<int>(std::cbrt(m_Params.N))}; 
    std::array<int, 2> particlesYZ {root, root};

    if ((root + 1)*particlesYZ[0]*particlesYZ[1] < m_Params.N)
        ++particlesYZ[0];
    if ((root + 1)*particlesYZ[0]*particlesYZ[1] < m_Params.N)
        ++particlesYZ[1];
    
    // Particles are evenly spaced along Y and Z axes, and spacing on x axis is either L/(root) or L/(root+1)
    std::array<float, 2> particleSpacing {m_Params.boxLength/particlesYZ[0], m_Params.boxLength/particlesYZ[1]};
    std::array<int, 3> pos {0, 0, 0}; // Location on lattice

    int extras {m_Params.N - root*particlesYZ[0]*particlesYZ[1]};
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
        if (pos[1] == particlesYZ[0]) {
            pos[1] = 0;
            ++pos[2];
        }

        // Shifting to account for the origin being the center of the box
        posXYZ[i] = pos[0] * m_Params.boxLength/(root + 1) + (m_Params.boxLength/(root + 1) / 2) - (m_Params.boxLength / 2);
        posXYZ[i + m_Params.N] = pos[1] * particleSpacing[0] + (particleSpacing[0] / 2) - (m_Params.boxLength / 2);
        posXYZ[i + 2*m_Params.N] = pos[2] * particleSpacing[1] + (particleSpacing[1] / 2) - (m_Params.boxLength / 2);

        // Updating m_Particles.cellIDs
        m_Particles.cellIDs[i] = getCellID(posXYZ[i], posXYZ[i + m_Params.N], posXYZ[i + 2*m_Params.N],
                                           m_Cells.cellsPerSide, m_Params.boxLength);

        ++i, ++pos[0];
    }

    // Once we run out of extras to fill, fill the rest of the X rows root # of particles
    while (i < m_Params.N) {
        if (pos[0] == root) {
            pos[0] = 0;
            ++pos[1];
        }
        if (pos[1] == particlesYZ[0]) {
            pos[1] = 0;
            ++pos[2];
        }

        posXYZ[i] = pos[0] * m_Params.boxLength / root + (m_Params.boxLength / root / 2) - (m_Params.boxLength / 2);
        posXYZ[i + m_Params.N] = pos[1] * particleSpacing[0] + (particleSpacing[0] / 2) - (m_Params.boxLength / 2);
        posXYZ[i + 2*m_Params.N] = pos[2] * particleSpacing[1] + (particleSpacing[1] / 2) - (m_Params.boxLength / 2);

        // Updating m_Particles.cellIDs
        m_Particles.cellIDs[i] = getCellID(posXYZ[i], posXYZ[i + m_Params.N], posXYZ[i + 2*m_Params.N],
                                           m_Cells.cellsPerSide, m_Params.boxLength);

        ++i, ++pos[0];
    }
    updateCellList();
}


void Simulator::populateRandom(float offset) {
    /*Fills the velocity list with a normal dist with mean 0 and variance = sqrt(Theta/m*)
    which is sqrt thermal energy divided by non-dimensional mass, the std of velocity components.*/
   std::vector<float>& velocities {m_Particles.velocitiesXYZ};
   rngKernelWrap(velocities.data(), std::size(velocities), false);
   std::transform(velocities.begin(),
                  velocities.end(),
                  velocities.begin(),
                  // Division by the non-dimensional mass currently 1
                  [this](float x) {return x * std::sqrt(m_Params.kT);}
                 );

   // Now filling randPositions to pull candidate positions from
   // Twice as big as necessary in case of overlap
   std::vector<float> randPositions(3 * m_Params.N * 2);
   rngKernelWrap(randPositions.data(), std::size(randPositions), true);
   std::transform(randPositions.begin(),
                  randPositions.end(),
                  randPositions.begin(),
                  [this](float x) {return x * m_Params.boxLength - (m_Params.boxLength / 2);}
                 );

   // Build out a dynamic temporary list
   std::vector<std::vector<int>> tempCellList(m_Cells.cellsTotal);

   // Particle pointer i, random number pointer j
   for (int i {0}, j {0}; i < m_Params.N; j += 3) {
        // refill the random numbers if we run out
        if (j >= (3 * m_Params.N * 2)) {
            rngKernelWrap(randPositions.data(), std::size(randPositions), true);
            std::transform(randPositions.begin(),
                           randPositions.end(),
                           randPositions.begin(),
                           [this](float x) {return x * m_Params.boxLength - (m_Params.boxLength / 2);}
                          );
            j = 0;
        }

        /*Get the prospective position and cellID, and loop through all neighbor cells
        in cell list, then every particle in every neighbor, and break if a particle
        is too close. In general the particle could be on the edge of the cell
        (i / 3) is the ID of the current particle. I write the random numbers straight
        to position array, and just overwrite them until it's a valid position*/
        std::vector<float>& pos {m_Particles.positionsXYZ};
        pos[i] = randPositions[j];
        pos[i + m_Params.N] = randPositions[j+1];
        pos[i + 2*m_Params.N] = randPositions[j+2];

        // Eventually make this function call with a float3 vector
        const int cellID {getCellID(pos[i], pos[i + m_Params.N], pos[i + 2*m_Params.N],
                                    m_Cells.cellsPerSide, m_Params.boxLength)};
        bool writePosition {true};
        for (int neighborIdx{cellID * 27}; neighborIdx < cellID * 27 + 27; ++neighborIdx) {
            for (const auto& particle : tempCellList[m_Cells.flatNeighborList[neighborIdx]]) {
                const float distance {getDistance(i, particle)};
                if (distance < (m_Tables.r_min + offset)) {
                    writePosition = false;
                    break;
                }
            }
            if (!writePosition) break;
        }
        if (writePosition) { 
            tempCellList[cellID].push_back(i);
            m_Particles.cellIDs[i] = cellID;
            i += 1;
        }
   }
   // Both things that should happen automatically after initial population
   updateCellList();
   zeroNetMomentum();
}


void Simulator::zeroNetMomentum() {
    /*Zeroes out the net momentum of the system by subtracting mass-weighted average
    velocity of N particles from each particle. (Currently mass=1, so just avg velocity)*/
    float sumVx {0}, sumVy {0}, sumVz {0};
    for (int i{0}; i < m_Params.N; ++i) {
        sumVx += m_Particles.velocitiesXYZ[i];
        sumVy += m_Particles.velocitiesXYZ[i + m_Params.N];
        sumVz += m_Particles.velocitiesXYZ[i + 2*m_Params.N];
    }

    const float meanVx {sumVx / m_Params.N};
    const float meanVy {sumVy / m_Params.N};
    const float meanVz {sumVz / m_Params.N};
    for (int i{0}; i < m_Params.N; ++i) {
        m_Particles.velocitiesXYZ[i] -= meanVx;
        m_Particles.velocitiesXYZ[i + m_Params.N] -= meanVy;
        m_Particles.velocitiesXYZ[i + 2*m_Params.N] -= meanVz;
    }
}


float Simulator::getDistance(int particleID1, int particleID2) const {
    /*Closely intertwined with populateRandom, since it depends on the particles being
    in the simulator data members, better design is to make it a pure function with float3
    structs. If the distance to the particle exceeds half the box length then it would be more
    appropriate to consider the interaction through the periodic boundary condition.
    Also it's ok if the di's are negative since they get squared.*/
    const std::vector<float>& pos {m_Particles.positionsXYZ};

    float dx {pos[particleID1] - pos[particleID2]};
    dx -= m_Params.boxLength * std::round(dx / m_Params.boxLength);

    float dy {pos[particleID1 + m_Params.N] - pos[particleID2 + m_Params.N]};
    dy -= m_Params.boxLength * std::round(dy / m_Params.boxLength);

    float dz {pos[particleID1 + 2 * m_Params.N] - pos[particleID2 + 2 * m_Params.N]};
    dz -= m_Params.boxLength * std::round(dz / m_Params.boxLength);

    return std::pow((dx*dx + dy*dy + dz*dz), 0.5);
}


void Simulator::updateCellList() {
    /*Rebuilds compact cellList, cellIndex, and particlesPerCell from particle cell IDs*/
    m_Cells.cellIndex[m_Cells.cellsTotal] = m_Params.N; // for traversal consistency

    // Counting particles in each cell
    for (int particle{0}; particle < m_Params.N; ++particle) {
        const int cellID {m_Particles.cellIDs[particle]};
        ++m_Cells.particlesPerCell[cellID];
    }
    // Prefix sum over counts to get cell start position in the compact cell list
    std::exclusive_scan(m_Cells.particlesPerCell.begin(), m_Cells.particlesPerCell.end(), m_Cells.cellIndex.begin(), 0);

    // We need to keep track of new location to write to not overwrite cellIndex
    std::vector<int> writeOffsets {m_Cells.cellIndex};
    for (int particle{0}; particle < m_Params.N; ++particle) {
        const int cellID {m_Particles.cellIDs[particle]}; // Cell particle i is in
        // Writes a particle into the compact cell list according to the correct cell index
        m_Cells.cellList[writeOffsets[cellID]++] = particle;
    }
}


void Simulator::fillLookupTables() {
    /*Fills the force and energy tables from r_min to r_max inclusive, and shifts the force
    and energy such that they go to zero at the cutoff r_max*/
    const float interval {(m_Tables.r_max - m_Tables.r_min) / (N_TABLE_ELEMENTS - 1)};
    const auto [energyAtRMax, forceAtRMax] {
        m_Tables.potential(m_Tables.r_max, m_Params.potentialArgs)
    };

    for (int i{0}; i < N_TABLE_ELEMENTS; ++i) {
        const float r {(i * interval) + m_Tables.r_min};
        const std::array<float, 2> v_f {m_Tables.potential(r, m_Params.potentialArgs)};
        m_Tables.forceTable[i] = v_f[1] - forceAtRMax;
        m_Tables.energyTable[i] = v_f[0] - energyAtRMax + (r - m_Tables.r_max) * forceAtRMax;
    }
}


void Simulator::fillNeighborList() {
    /*Fills a flat list containing the 27 neighbor cells for each center cellID in order,
    including the center cell, which is always the first in the corresponding list. Such
    that the 27*ith element = i.*/
    const int cps {m_Cells.cellsPerSide};
    const int cellsPerLevel {cps * cps};

    for (int cellID{0}; cellID < m_Cells.cellsTotal; ++cellID) {

        const int x_cell {(cellID % cellsPerLevel) % cps};
        const int y_cell {(cellID % cellsPerLevel) / cps};
        const int z_cell {cellID / cellsPerLevel};

        // First entry is always the cell itself
        int index{0};
        m_Cells.flatNeighborList[cellID * 27 + index] = cellID;
        ++index;

        for (int i{-1}; i <= 1; ++i) {
            for (int j{-1}; j <= 1; ++j) {
                for (int k{-1}; k <= 1; ++k) {

                    // Already stored the center cell
                    if (i == 0 && j == 0 && k == 0)
                        continue;

                    const int nx {(x_cell + i + cps) % cps};
                    const int ny {(y_cell + j + cps) % cps};
                    const int nz {(z_cell + k + cps) % cps};
                    const int neighborID {nx + ny * cps + nz * cellsPerLevel};
                    
                    m_Cells.flatNeighborList[cellID * 27 + index] = neighborID;
                    ++index;
                }
            }
        }
    }
}


void Simulator::integrate(const bool writeEnergy) {
    integrateKernelWrapper(this, writeEnergy);
}
