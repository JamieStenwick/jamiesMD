#include "DeviceParams.cuh"
#include "Simulator.hpp"


DeviceParams::DeviceParams(Simulator* Sim) :
    /*Container for misc fundamental type parameters needed for integration*/
    cellsPerSide {Sim->m_Cells.cellsPerSide},
    cellsTotal {cellsPerSide * cellsPerSide * cellsPerSide},
    boxLength {Sim->m_Params.boxLength},
    dt {Sim->m_Params.dt},
    timeSteps {Sim->m_Params.timeSteps},
    // mass=1, gamma=1 in both of these
    c {expf(-Sim->m_Params.dt)}, // mass=1, gamma=1 in exp
    sigma {sqrtf(Sim->m_Params.kT * (-expm1f(-2 * Sim->m_Params.dt)))} // expm1 for numerical stability
    {}
