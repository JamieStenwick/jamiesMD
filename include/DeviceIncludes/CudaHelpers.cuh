#pragma once


// float3 operator overloads
__device__ __forceinline__ float3 operator+(const float3& arr1, const float3& arr2) {
    /*Helper for float3 operations*/
    return {arr1.x + arr2.x, arr1.y + arr2.y, arr1.z + arr2.z};
}


__device__ __forceinline__ float3 operator*(const float3& arr, const float scalar) {
    /*Helper for float3 operations*/
    return {arr.x * scalar, arr.y * scalar, arr.z * scalar};
}


__device__ __forceinline__ float3 getDistance(const float3& pos1, const float3& pos2, const float boxLength) {
    /*Takes two particles' positions in float3, and returns a float4 with the .x .y .z .w
    attributes equal to dx, dy, dz, and dr respectively. Handles periodic boundary conditions.*/
    float dx {pos1.x - pos2.x};
    dx -= boxLength * round(dx / boxLength);

    float dy {pos1.y - pos2.y};
    dy -= boxLength * round(dy / boxLength);

    float dz {pos1.z - pos2.z};
    dz -= boxLength * round(dz / boxLength);

    return {dx, dy, dz};
}


__device__ __forceinline__ int getCellID(float3 pos, int cellsPerSide, float boxLength) {
    /*(0, 0, 0) is center of box but cellID 0 still starts in negative most corner and
    increases first along +x, then +y, then +z Periodic boundary condition is a half length
    around the box. Returning -1 sets error flag on device side.*/
    if (fabsf(pos.x) > (boxLength) || fabsf(pos.y) > (boxLength) || fabsf(pos.z) > (boxLength)) {
        return -1;
    }

    // Shift particle back to original box if in periodic boundary layer, boundary is half open for consistency
    if (pos.x >= boxLength / 2.0f) pos.x -= boxLength;
    if (pos.x < -boxLength / 2.0f) pos.x += boxLength;

    if (pos.y >= boxLength / 2.0f) pos.y -= boxLength;
    if (pos.y < -boxLength / 2.0f) pos.y += boxLength;

    if (pos.z >= boxLength / 2.0f) pos.z -= boxLength;
    if (pos.z < -boxLength / 2.0f) pos.z += boxLength;

    // Shift center coordinates to corner before calculating cell index, and clamp max cellID cuz of fp math
    float cellLength {boxLength / cellsPerSide};
    const int x_cell {min(cellsPerSide - 1, static_cast<int>((pos.x + boxLength / 2.0f) / cellLength))};
    const int y_cell {min(cellsPerSide - 1, static_cast<int>((pos.y + boxLength / 2.0f) / cellLength))};
    const int z_cell {min(cellsPerSide - 1, static_cast<int>((pos.z + boxLength / 2.0f) / cellLength))};

    return x_cell + y_cell*cellsPerSide + z_cell*cellsPerSide*cellsPerSide;
}


// Templates for now are unecessary, we'll see about the future though
template <typename T>
__device__ __forceinline__ float3 getVectorXYZ(const T* array, const int idx, const int N) {
    /*Returns a float3 of the specified idx from a flattened XYZ array, array has length 3N*/
    float3 vector;
    vector.x = array[idx];
    vector.y = array[idx + N];
    vector.z = array[idx + 2*N];

    return vector;
}


template <typename T>
__device__ __forceinline__ void writeVectorXYZ(const float3& vector, T* array, const int idx, const int N) {
    /*Writes a float3 of the specified idx to a flattened XYZ array, array has length 3N*/
    array[idx] = vector.x;
    array[idx + N] = vector.y;
    array[idx + 2*N] = vector.z;
}


template <typename T>
__device__ __forceinline__ void writeVectorXYZ(T* arr1, const int idx1, const int N1,
                                               T* arr2, const int idx2, const int N2) {
    /*Writes a vector from arr2 with idx2 and length 3N2 to arr1 with idx1 and length 3N1*/
    arr1[idx1] = arr2[idx2];
    arr1[idx1 + N1] = arr2[idx2 + N2];
    arr1[idx1 + 2*N1] = arr2[idx2 + 2*N2];
}
