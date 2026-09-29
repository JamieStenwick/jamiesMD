#include <SimUtilities.hpp>
#include <fstream>
#include <exception>
#include <cmath>


void writePositions(const float* positionsXYZ, const int N, const bool newFile) {
    /*Writes a positions vector x, y, z to positions.txt, appending if frame > 1*/
    if (newFile) {
        std::ofstream outfile{"positions.txt"};
        if (!outfile) {throw std::runtime_error("outfile could not be opened");}

        for (int i{0}; i < N; i++) {
            outfile << positionsXYZ[i] << ' ' << positionsXYZ[i + N] << ' ' << positionsXYZ[i + 2*N] << '\n';
        }
    }
    else {
        std::ofstream outfile{"positions.txt", std::ios::app};
        if (!outfile) {throw std::runtime_error("outfile could not be opened");}

        for (int i{0}; i < N; i++) {
            outfile << positionsXYZ[i] << ' ' << positionsXYZ[i + N] << ' ' << positionsXYZ[i + 2*N] << '\n';
        }
    }
}


void writeEnergies(const float* potentialEnergies,
                   const float* velocitiesXYZ,
                   const float* forces,
                   const int N,
                   const float dt,
                   const bool newFile) {
    /*Writes the system's potential, kinetic, and total energy to energies.txt*/
    float U {0};
    float K {0};
    for (int i{0}; i < N; ++i) {
        U += potentialEnergies[i];

        float vx {velocitiesXYZ[i]};
        float vy {velocitiesXYZ[i + N]};
        float vz {velocitiesXYZ[i + 2*N]};
        if (!newFile) {
            const float fx {forces[i]}, fy {forces[i + N]}, fz {forces[i + 2*N]};
            vx += fx * dt / 2;
            vy += fy * dt / 2;
            vz += fz * dt / 2;
        }
        K += vx*vx + vy*vy + vz*vz;
    }
    U /= 2.0; // Double counted
    K /= 2.0; // K = 0.5mv^2, mass=1

    const float T {U + K};
    if (newFile) {
        std::ofstream outfile{"energies.txt"};
        if (!outfile) {throw std::runtime_error("outfile could not be opened");}
        outfile << U << ' ' << K << ' ' << T << '\n';
    }
    else {
        std::ofstream outfile{"energies.txt", std::ios::app};
        if (!outfile) {throw std::runtime_error("outfile could not be opened");}
        outfile << U << ' ' << K << ' ' << T << '\n';
    }
}


int getCellID(float x, float y, float z, int cellsPerSide, float boxLength) {
    /*Takes a position and the cell parameters for a box and returns the cellID, throws exception
    if outside of PBC layer, cellID's start at 0 in the corner of the box, not the center (origin)*/
    if (std::abs(x) > (boxLength) || std::abs(y) > (boxLength) || std::abs(z) > (boxLength))
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
    int x_cell {std::min(cellsPerSide - 1, static_cast<int>((x + boxLength / 2.0f) / cellLength))};
    int y_cell {std::min(cellsPerSide - 1, static_cast<int>((y + boxLength / 2.0f) / cellLength))};
    int z_cell {std::min(cellsPerSide - 1, static_cast<int>((z + boxLength / 2.0f) / cellLength))};

    return x_cell + y_cell*cellsPerSide + z_cell*cellsPerSide*cellsPerSide;
}


double getRmin (const std::vector<float>& args, const float thermalEnergy) {
    /*Analytically determine the r_min cutoff for the simulation by finding U(r_min)
    such that at thermal equilibrium, the chances of finding a particle pair at that
    separation is extremely low, according to the Boltzmann weights this isn't to say
    it's impossible for it to be crossed though. U(r_min) - U(r_eq) = 20*Theta (thermal energy)
    is a good start, since e^-20 ~ 2*10^-9. LJ-specific for analytically finding the value.*/
    if (args[0] <= 0)
        throw std::runtime_error("LJ parameter epsilon must be greater than 0");

    const double potentialCutoff {20*thermalEnergy - args[0]};
    const double y {0.5 * (1 + std::sqrt(1 + 4 * (potentialCutoff / (4 * args[0]))))};

    return args[1] / (std::pow(y, 1.0/6.0));
}


double getRmax (const float sigma) {
    /*Currently LJ specific, since it's dependent on sigma, though I believe it's better
    than choosing an arbitrarily small force then using a root solver, for the LJ case at least.*/
    return 3 * sigma;
}
