#include "DeviceLookupTables.cuh"
#include "SimulatorDataStructs.hpp"


DeviceLookupTables::DeviceLookupTables(SimTables& Tables) {
    m_data.tableElements = Tables.N;
    m_data.r_min = Tables.r_min;
    m_data.r_max = Tables.r_max;
    m_data.maxForce = Tables.forceTable[0];
    m_data.stepSize = (m_data.r_max - m_data.r_min) / (m_data.tableElements - 1);

    cudaMalloc(&m_data.forceTable, m_data.tableElements * sizeof(float));
    cudaMalloc(&m_data.energyTable, m_data.tableElements * sizeof(float));
}

DeviceLookupTables::~DeviceLookupTables() {
    cudaFree(m_data.forceTable);
    cudaFree(m_data.energyTable);
}


void DeviceLookupTables::copyFromHost(SimTables& hostTables) const {
    /*Copies the force tables owned by a simulator to gpu for integration*/
    cudaMemcpy(m_data.forceTable, hostTables.forceTable.data(),
                m_data.tableElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_data.energyTable, hostTables.energyTable.data(),
                m_data.tableElements * sizeof(float), cudaMemcpyHostToDevice);
}
