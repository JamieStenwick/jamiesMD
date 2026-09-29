#pragma once
#include <vector>


void writePositions(const float* positionsXYZ, const int N, const bool newFile);
void writeEnergies(const float* potentialEnergies, const float* velocitiesXYZ, const float* forces,
                   const int N, const float dt, const bool newFile);

int getCellID(float x, float y, float z, int cellsPerSide, float boxLength);

double getRmin (const std::vector<float>& args, const float thermalEnergy);

double getRmax (const float sigma);
