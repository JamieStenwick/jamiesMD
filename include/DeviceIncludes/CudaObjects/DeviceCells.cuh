#pragma once


struct SimCells;


struct CellData {
    int* cellList {nullptr};
    int* particlesPerCell {nullptr};
    int* cellIndex {nullptr};
    int* cellOffsets {nullptr};
    int* flatNeighborList {nullptr};
};


class DeviceCells {
    /*Unique owner of the gpu resources for a Cell Data struct
    Initializes the struct and handles allocation and destruction of gpu memory*/
private:
    CellData m_data {};

public:
    explicit DeviceCells(int N, int cells);
    ~DeviceCells();

    DeviceCells(const DeviceCells&) = delete;
    DeviceCells& operator=(const DeviceCells&) = delete;

    CellData data() const {
        return m_data;
    }

    // Change to just a reference to SimCell struct when you make it
    void copyFromHost(SimCells& hostCells) const;
};
