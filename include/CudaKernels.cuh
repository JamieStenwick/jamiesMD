#pragma once

class Simulator;

void popRandKernelWrap(float* output, const std::size_t N, const float boxLength);

void integrateKernelWrapper(Simulator* Sim);
