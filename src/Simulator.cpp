#include <cstdlib>
#include <numbers>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <exception>
#include <array>
#include <iterator>
#include <numeric>
#include "Simulator.hpp"
#include "CudaKernels.cuh"
#include <iostream>
#include <boost/math/tools/roots.hpp>


// We are deriving r cutoffs from the maximum and minimum displacements i would want to allow
// from conservative forces
constexpr int N_TABLE_ELEMENTS {50000};
constexpr double MAX_FORCE_DISPLACEMENT{0.01};
constexpr double MIN_FORCE_DISPLACEMENT{1e-6};

// Declaring in here for now, will probably move to separate functions file when we get enough non-Simulator functions
double getRmin (PotentialFuncPtr potential, const std::vector<float>& args, const double maxForce) {
    /*Do root finding according to desired maximum force to find acceptable rmin. Force rather
    than energy is technically what matters since that will produce poorly defined integration
    behavior. CURRENTLY LENNARD JONES SPECIFIC */
    auto f {
        [potential, args, maxForce](double r) {
        return std::abs(potential(r, args)[1]) - maxForce;
        }
    };

    // Make this a variable because actual iters are written into it
    std::uintmax_t maxIter = 100;
    
    // Known upper bound for LJ, then scale back lower bound until we cross maxForce
    const double upperBracket {std::pow(2.0, 1.0/6) * args[1]};
    double lowerBracket {0.5 * upperBracket};
    while (f(lowerBracket) < 0.0) {
        lowerBracket *= 0.5;
    }

    auto result = boost::math::tools::toms748_solve(
        f,
        lowerBracket,
        upperBracket,
        boost::math::tools::eps_tolerance<double>(40), // Bits of precision (double has 53 for ref)
        maxIter
    );

    return 0.5 * (result.first + result.second);
}


double getRmax (PotentialFuncPtr potential, const std::vector<float>& args, const double minForce) {
    /*Takes a potential, args, and a tolerance as input and returns the radius at which the potential
    crosses the tolerance in it's monotonically decreasing tail.*/
    auto f {
        [potential, args, minForce](double r) {
        return std::abs(potential(r, args)[1]) - std::min(minForce, 0.02);
        }
    };

    double tailStart {-1.0};
    double rCurrent {2.0}; // Valid for all non-overlapping potentials

    /*March forward by a multiplicative factor, accepting a tail if it's been monotonically
    decreasing AND less than tolerance for two times the radius at the start of the tail,
    resets if it starts increasing or leaves tolerance.*/
    constexpr double growth {1.05};
    constexpr double tailDistanceFactor {2.0};
    while (true) {
        const double rNext {rCurrent * growth};
        const double fCurrent {f(rCurrent)};
        const double fNext {f(rNext)};

        if (fNext <= 0.0 && fNext < fCurrent) {
            if (tailStart < 0.0)
                tailStart = rCurrent;

            if (rNext >= 2.0 * tailStart)
                break;
        }
        else {
            tailStart = -1.0;
        }
        rCurrent = rNext;
    }

    std::uintmax_t maxIter = 100;

    auto result = boost::math::tools::toms748_solve(
        f,
        tailStart, // Lower bracket
        rCurrent, // Upper bracket
        boost::math::tools::eps_tolerance<double>(40), // Bits of precision (double has 53 for ref)
        maxIter
    );

    return 0.5 * (result.first + result.second);
}


// Clean up having to send x, y, and z if possible
int getCellID(float x, float y, float z, int cellsPerSide, float boxLength) {
    // (0, 0, 0) is center of box but cellID 0 still starts in negative most corner
    // and increases first along +x, then +y, then +z
    if (std::abs(x) > (1.5*boxLength) || std::abs(y) > (1.5*boxLength) || std::abs(z) > (1.5*boxLength))
        throw std::runtime_error("Particle has left simulation box and periodic boundary conditions");

    // Shift particle back to original box if in periodic boundary layer, boundary is half open for consistency
    if (x >= boxLength / 2.0f) x -= boxLength;
    if (x < -boxLength / 2.0f) x += boxLength;

    if (y >= boxLength / 2.0f) y -= boxLength;
    if (y < -boxLength / 2.0f) y += boxLength;

    if (z >= boxLength / 2.0f) z -= boxLength;
    if (z < -boxLength / 2.0f) z += boxLength;

    // Shift center coordinates to corner before calculating cell index
    float cellLength {boxLength / cellsPerSide};
    int x_cell {static_cast<int>((x + boxLength / 2.0f) / cellLength)};
    int y_cell {static_cast<int>((y + boxLength / 2.0f) / cellLength)};
    int z_cell {static_cast<int>((z + boxLength / 2.0f) / cellLength)};

    return x_cell + y_cell*cellsPerSide + z_cell*cellsPerSide*cellsPerSide;
}


void writePositions(const float* positionsXYZ, const int N, const int frame, const bool newFile) {
    // Remember to try{} this, also will open in binary mode for during runtime
    if (newFile) {
        std::ofstream outfile{"positions.txt"};
        // if (!outfile) {throw std::runtime_error("outfile could not be opened");}

        for (int i{0}; i < N; i++) {
            outfile << positionsXYZ[i] << ' ' << positionsXYZ[i + N] << ' ' << positionsXYZ[i + 2*N] << '\n';
        }
    }
    else {
        std::ofstream outfile{"positions.txt", std::ios::app};
        // if (!outfile) {throw std::runtime_error("outfile could not be opened");}

        for (int i{0}; i < N; i++) {
            outfile << positionsXYZ[i] << ' ' << positionsXYZ[i + N] << ' ' << positionsXYZ[i + 2*N] << '\n';
        }
    }
}


Simulator::Simulator(const SimParams& Params, const PotentialFuncPtr potential) : // Using vector to hold all possible other args for now, will come back to this
        m_Params {Params},
        m_Tables {N_TABLE_ELEMENTS, potential, Params.potentialArgs,
                  MIN_FORCE_DISPLACEMENT, MAX_FORCE_DISPLACEMENT, Params.dt},
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
   // First filling the velocities randomly, will need normal dist later
   popRandKernelWrap(m_Particles.velocitiesXYZ.data(), std::size(m_Particles.velocitiesXYZ), 2.0f);

   // Now filling randPositions to pull candidate positions from
   // Twice as big as necessary in case of overlap
   std::vector<float> randPositions(3 * m_Params.N * 2);
   popRandKernelWrap(randPositions.data(), std::size(randPositions), m_Params.boxLength);

   // Build out a dynamic temporary list
   std::vector<std::vector<int>> tempCellList(m_Cells.cellsTotal);

   // Particle pointer i, random number pointer j
   for (int i {0}, j {0}; i < m_Params.N; j += 3) {
        // refill the random numbers if we run out
        if (j >= (3 * m_Params.N * 2)) {
            popRandKernelWrap(randPositions.data(), std::size(randPositions), m_Params.boxLength);
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

        // Need to clean up this fucking getCellID function it's ugly as shit
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
   updateCellList();
}


float Simulator::getDistance(int particleID1, int particleID2) const {
    /*If the distance to the particle exceeds half the box length then it would be more
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

    /* Useful pattern for accessing cell list
    for (int i{0}; i < cells; ++i) {
        std::cout << "Cell " << i << " has particles:" << '\n';
        for (int j{m_cellIndex[i]}; j < m_cellIndex[i + 1]; ++j) {
            std::cout << m_cellList[j] << '\n';
        }
    }
    std::cout << cells << " total cells" << '\n';
    */
}


void Simulator::fillLookupTables() {
    // Remember, N equally space elements means a spacing of (range) / (N - 1)
    const float interval {(m_Tables.r_max - m_Tables.r_min) / (N_TABLE_ELEMENTS - 1)};

    for (int i{0}; i < N_TABLE_ELEMENTS; ++i) {
        const float r {(i * interval) + m_Tables.r_min};
        const std::array<float, 2> v_f {m_Tables.potential(r, m_Params.potentialArgs)};
        m_Tables.forceEnergyTable[i] = v_f[1];
        m_Tables.forceEnergyTable[i + N_TABLE_ELEMENTS] = v_f[0];
    }
}


void Simulator::fillNeighborList() {
    /*Takes a cell ID and number of cells per dimension as input, and returns an array
    of the 26 neighbor cell IDs + the cell ID passed, including the periodic boundary
    conditions for cells on the edge of the simulation box.*/
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


void Simulator::integrate() {
    integrateKernelWrapper(this);
}
