// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DATABASE_POTRF_CLUSTER_DB_CUH
#define CUSOLVERDX_DATABASE_POTRF_CLUSTER_DB_CUH

#include "cusolverdx/detail/type_enum.hpp"

namespace cusolverdx::detail::potrf {

    // Tuned from build-bench/potrf_cluster_{float,double}_{real,complex}.sm{90,100}.out (Arch 900/1000).
    constexpr inline __device__ __host__ unsigned suggested_cluster_tile_size(type_enum T, unsigned N, unsigned BPC, int Arch) {
        if (BPC == 1) {
            return N;
        }
        if (Arch == 900 || Arch == 1000) {
            if (T == type_enum::real_f32) {
                if (N <= 128) {
                    return 32;
                }
                if (BPC == 2) {
                    if (N <= 256) {
                        return 64;
                    }
                    if (N == 288) {
                        return 96;
                    }
                    if (N <= 320) {
                        return 64;
                    }
                    if (N <= 384) {
                        return 96;
                    }
                    return 64;
                }
                if (BPC == 4) {
                    if (Arch == 1000 && N == 256) {
                        return 64;
                    }
                    if (N <= 256) {
                        return 96;
                    }
                    if (Arch == 1000 && N == 288) {
                        return 96;
                    }
                    if (N <= 320) {
                        return 48;
                    }
                    if (N == 352) {
                        return (Arch == 900) ? 32u : 96u;
                    }
                    if (N <= 384) {
                        return 96;
                    }
                    if (N <= 416) {
                        return 48;
                    }
                    if (N == 480) {
                        return 96;
                    }
                    if (N <= 512) {
                        return 64;
                    }
                    return 96;
                }
                if (BPC == 8) {
                    if (Arch == 1000 && N == 256) {
                        return 64;
                    }
                    if (N <= 256) {
                        return 32;
                    }
                    if (Arch == 1000 && N == 320) {
                        return 64;
                    }
                    if (Arch == 1000 && N == 352) {
                        return 96;
                    }
                    if (N <= 352) {
                        return 48;
                    }
                    if (Arch == 1000 && N == 512) {
                        return 64;
                    }
                    return 96;
                }
            }
            if (T == type_enum::real_f64 || T == type_enum::complex_f32) {
                if (N <= 160) {
                    return 48;
                }
                if (BPC == 2) {
                    if (N == 192) {
                        return (Arch == 900) ? 64u : 48u;
                    }
                    if (N <= 224) {
                        return 32;
                    }
                    if (N <= 256) {
                        return 64;
                    }
                    if (N == 288) {
                        return 96;
                    }
                    return 32;
                }
                if (BPC == 4) {
                    if (N <= 224) {
                        return (Arch == 900 || N == 192) ? 48u : 32u;
                    }
                    if (N <= 256) {
                        return 64;
                    }
                    if (N <= 288) {
                        return 48;
                    }
                    if (N <= 320) {
                        return (Arch == 900) ? 64u : 32u;
                    }
                    if (N <= 384) {
                        return 96;
                    }
                    if (Arch == 1000 && N == 416) {
                        return 48;
                    }
                    return 64;
                }
                if (BPC == 8) {
                    if (N <= 224) {
                        return 48;
                    }
                    if (Arch == 1000 && N == 256) {
                        return 64;
                    }
                    if (N <= 288) {
                        return 48;
                    }
                    if (Arch == 1000 && N == 320) {
                        return 64;
                    }
                    if (N <= 416) {
                        return 48;
                    }
                    if (Arch == 1000 && N == 448) {
                        return 64;
                    }
                    if (N <= 480) {
                        return 96;
                    }
                    if (N <= 512) {
                        return 64;
                    }
                    return 96;
                }
            }
            if (T == type_enum::complex_f64) {
                if (N <= 160) {
                    return 32;
                }
                if (BPC == 2) {
                    if (N == 192) {
                        return (Arch == 900) ? 48u : 64u;
                    }
                    return 32;
                }
                if (BPC == 4) {
                    if (N <= 192) {
                        return 48;
                    }
                    if (N == 256) {
                        return (Arch == 900) ? 32u : 64u;
                    }
                    if (N == 288) {
                        return 48;
                    }
                    return 32;
                }
                if (BPC == 8) {
                    if (Arch == 1000 && N <= 224) {
                        return 48;
                    }
                    if (N <= 256) {
                        return 32;
                    }
                    if (N <= 288) {
                        return 48;
                    }
                    if (N <= 352) {
                        return 64;
                    }
                    if (Arch == 1000 && N <= 384) {
                        return 64;
                    }
                    if (N <= 384) {
                        return 48;
                    }
                    return 32;
                }
            }
        }
        return 64;
    }

    // Suggested block dim (threads per CTA) from potrf_cluster_*.sm{90,100}.out.
    constexpr inline __device__ __host__ dim3 suggested_cluster_block_dim(type_enum T, unsigned N, unsigned TileSize, unsigned BPC, int Arch) {
        if (Arch != 900 && Arch != 1000) {
            return 128;
        }

        if (TileSize <= 32) {
            return 128;
        }

        if (BPC == 2) {
            if (TileSize >= 96) {
                return 256;
            }
            if (Arch == 900 && TileSize >= 64) {
                return 256;
            }
            return 128;
        }

        if (Arch == 900 && T == type_enum::real_f32 && BPC == 4 && TileSize >= 96 && N >= 384) {
            return 256;
        }

        if (T == type_enum::complex_f32 && BPC >= 4 && TileSize >= 96) {
            return 256;
        }

        return 128;
    }

} // namespace cusolverdx::detail::potrf

#endif // CUSOLVERDX_DATABASE_POTRF_CLUSTER_DB_CUH
