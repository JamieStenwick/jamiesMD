/*This software contains source code provided by NVIDIA Corporation.*/
#include <iostream>
#include <stdexcept>
#include <array>
#include <cmath>
#include <cstdlib>
#include "Simulator.hpp"


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
			std::cout << "Usage: <executable> <dt> <vol_frac> <kTs>"
                         "<N> <timeTotal> <frames> [potential_args]" << '\n';
		return 1;
	}

    // Runtime errors can currently only happen during integration
    try {
        SimParams Params(argc, argv);
        Simulator Sim{Params, &lJonesPotential};

        Sim.populateRandom(0.1f);
        Sim.integrate(true);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return 1;
    }
}
