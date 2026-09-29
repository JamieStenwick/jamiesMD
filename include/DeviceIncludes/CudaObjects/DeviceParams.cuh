#pragma once


class Simulator;


struct DeviceParams {
    /*Container for the relevant quantities specifically for integration*/
    explicit DeviceParams(Simulator* Sim);

    int cellsPerSide;
    int cellsTotal;
    float boxLength;
    float dt;
    int timeSteps; // Unnecessary for now, since looping is controlled outside of kernel
    float c;
    float sigma;
};
