#include <iostream>
#include <array>
#include <cmath>
#include <cstdlib>
#include "Simulator.hpp"
/* Jamie Stenwick; as simple of a molecular dynamics simulation as one can get.
Assumptions / Current Conditions:
Lennard-Jones Potential, V(r) = 4*eps[(sigma/r)^12 - (sigma/r)^6]
Passed Potential function applies to all particle interactions
All radii = 1 (reference length)
Reference energy = kT_ref
All mass = 1 (reference mass)
The 3 indpendent units we will choose are length, energy, and mass
L_0, E_0, M_0 are the reference length, energy, and mass respectively, in our simulation
we take L_0 = a, a = particle radius, E_0 = kT_ref, M_0 = m_0, m_0 = particle mass
Non-dimensional Drag = 1 (not a reference value, gamma * L_0 / sqrt(M_0 * E_0))
r distance coordinate is center-center in radii
Translational DOF only
Cell list
*/


std::array<float, 2> lJonesPotential(float radius, const std::vector<float>& args) {
    using std::pow;
    float epsilon {args[0]};
    float sigma {args[1]};
    float V {static_cast<float>(4 * epsilon * (pow(sigma / radius, 12) - pow(sigma / radius, 6)))};
    float F {static_cast<float>(24 * epsilon * (2 * (pow(sigma, 12) / pow(radius, 13)) - (pow(sigma, 6) / pow(radius, 7))))};

    return {V, F};
}


int main(int argc, char* argv[]) {
    /*Entry point for simulation*/
    constexpr int requiredArgs = 6;
    if (argc < requiredArgs + 1) {
			std::cout << "Usage: <executable> <dt> <vol_frac> <temp>"
                         "<N> <t_steps> <frames> [potential_args]" << '\n';
		return 1;
	}

    SimParams Params(argc, argv); // Verified constructor
    // std::cout << Params;  Verified

    Simulator Sim{Params, &lJonesPotential}; // Compiles, will test further on implementing getRmin
    // Sim.populateLattice(); // Verified
    Sim.populateRandom(0.1f);

    Sim.integrate();

    // Function Testing
    // std::cout << getCellID(89.42, 69.67, 102.8, 5.623, 170.3) << '\n'; // Seems to work well enough
    // std::vector<int> testList {getNeighborList(7845, 34)};
    // for (const auto& cell : testList) {std::cout << cell << '\n';} Seems to work well enough

    return 0;
}
