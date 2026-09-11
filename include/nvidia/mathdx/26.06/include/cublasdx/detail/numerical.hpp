// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_NUMERICAL_HPP
#define CUBLASDX_DETAIL_NUMERICAL_HPP

namespace cublasdx {
    namespace detail {
        // Find lower integer bound on square root of argument
        // such that
        // v = lower_int_sqrt(x)
        // v * v <= x
        // (v + 1) * (v + 1) >= x
        constexpr int lower_bound_int_sqrt(int v) {
            int low  = 0;
            int high = v;

            while (low != high) {
                auto mid = (low + high + 1) / 2;
                if (v / mid < mid) {
                    high = mid - 1;
                } else {
                    low = mid;
                }
            }

            return low;
        }

        template<int N>
        constexpr int closest_multiple_of(int value) {
            return ((value + (N - 1)) / N) * N;
        }

        constexpr int closest_next_power_of_two(int v) {
            // source: https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return v;
        }

        template<typename T1, typename T2>
        constexpr T1 round_up(T1 value, T2 alignment) {
            const auto unified_alignment = static_cast<T1>(alignment);
            return ((value + (unified_alignment - 1)) / unified_alignment) * unified_alignment;
        }
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_NUMERICAL_HPP