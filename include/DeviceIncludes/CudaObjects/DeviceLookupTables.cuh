#pragma once


struct SimTables;


struct LookupTableData {
    float* forceTable {nullptr};
    float* energyTable {nullptr};
    int tableElements;

    float r_min;
    float r_max;
    float maxForce;
    float stepSize;
};


class DeviceLookupTables {
    /*Unique owner of the gpu resources for a LookupTableData struct.
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    LookupTableData m_data {};

public:
    explicit DeviceLookupTables(SimTables& Tables);
    ~DeviceLookupTables();

    DeviceLookupTables(const DeviceLookupTables&) = delete;
    DeviceLookupTables& operator=(const DeviceLookupTables&) = delete;

    LookupTableData data() const {
        return m_data;
    }

    void copyFromHost(SimTables& hostTables) const;
};
