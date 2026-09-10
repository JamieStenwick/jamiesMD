#include <iostream>
#include <array>
#include <cmath>
#include <cstdlib>
#include "Simulator.hpp"
/* Jamie Stenwick; as simple of a molecular dynamics simulation as one can get.
Assumptions / Current Conditions:
Lennard-Jones Potential, V(r) = 4*eps[(sigma/r)^12 - (sigma/r)^6]
All radii = 1
All drag = 1
Cell list
*/

/*
struct SimParams {
    Ordered doubles to reduce padding, constructor takes the vector of command line args
    and initializes the members with the values, MUST pass cmd-line args in same order as
    struct constructor. strtod and strtol require nullptr arg for some reason
    SimParams(char* args[]) :
        dt {std::strtod(args[1], nullptr)},
        vol_frac {std::strtod(args[2], nullptr)},
        temp {std::strtod(args[3], nullptr)},
        epsilon {std::strtod(args[4], nullptr)},
        sigma {std::strtod(args[5], nullptr)},

        // long to int safe conversion
        N {std::strtol(args[6], nullptr, 10)},
        t_steps {std::strtol(args[7], nullptr, 10)},
        frames {std::strtol(args[8], nullptr, 10)}
        {}
    double dt;
    double vol_frac;
    double temp;
    double epsilon;
    double sigma;
    int N;
    int t_steps;
    int frames;
};
*/

// Params testing
std::ostream& operator<<(std::ostream& out, SimParams& p) {
        return out << p.N << p.t_steps << p.dt << p.frames << p.vol_frac << p.temp << p.epsilon << p.sigma;
    }


std::array<double, 2> lJonesPotential(double radius, double epsilon, double sigma) {
    using std::pow;
    double V {4 * epsilon * (pow(sigma / radius, 12) - pow(sigma / radius, 6))};
    double F {48 * epsilon * ((pow(sigma, 12) / pow(radius, 13)) - (pow(sigma, 6) / pow(radius, 7)))};

    return {V, F};
}


int main(int argc, char* argv[]) {
    /*Entry point for simulation*/
    if (argc != 9) {
			std::cout << "Usage: .\\main <dt> <vol_frac> <temp> <epsilon>"
                         "<sigma> <N> <t_steps> <frames>" << '\n';
		return 1;
	}

    const SimParams Params(argv); // Verified constructor
    // std::cout << Params;  Verified

    Simulator Sim{Params}; // Verified constructor
    // Sim.setPositionsLattice(); Verified
    // Sim.writePositions(); Need to implement exception throw on filesystem error works


    return 0;
}
