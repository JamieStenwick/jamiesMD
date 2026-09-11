// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_SELECT_type
#define CURANDDX_DETAIL_SELECT_type

// CTK include header
#include <curand_kernel.h>
//#include <curand_mtgp32_host.h>
// include MTGP pre-computed parameter sets
//#include <curand_mtgp32dc_p_11213.h>
#include "rand_philox.hpp"
#include "rand_pcg.hpp"

namespace curanddx {
    namespace detail {

        template<generator Value, unsigned int rounds = 10>
        struct select_generator_type {
            using state_type            = void;
            using result_type           = void;
            using offset_type           = void;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<xorwow> {
            using state_type            = curandStateXORWOW_t;
            using result_type           = unsigned int;
            using offset_type           = unsigned long long;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<mrg32k3a> {
            using state_type            = curandStateMRG32k3a_t;
            using result_type           = unsigned int;
            using offset_type           = unsigned long long;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<unsigned int rounds>
        struct select_generator_type<philox4_32, rounds> {
            using state_type            = struct philox4x32_native_state<rounds>;
            using result_type           = uint4;
            using offset_type           = unsigned long long;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<mtgp32> {
            using state_type            = curandStateMtgp32_t;
            using result_type           = unsigned int;
            using offset_type           = void;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<pcg> {
            using state_type            = struct pcg_state;
            using result_type           = unsigned int;
            using offset_type           = unsigned long long;
            using direction_vector_type = void;
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<sobol32> {
            using state_type            = curandStateSobol32_t;
            using result_type           = unsigned int;
            using offset_type           = unsigned int;
            using direction_vector_type = unsigned int[32]; // curandDirectionVectors32_t
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<scrambled_sobol32> {
            using state_type            = curandStateScrambledSobol32_t;
            using result_type           = unsigned int;
            using offset_type           = unsigned int;
            using direction_vector_type = unsigned int[32];
            using scrambled_const_type  = unsigned int;
        };
        template<>
        struct select_generator_type<sobol64> {
            using state_type            = curandStateSobol64_t;
            using result_type           = unsigned long long;
            using offset_type           = unsigned long long;
            using direction_vector_type = unsigned long long[64]; // curandDirectionVectors64_t
            using scrambled_const_type  = void;
        };
        template<>
        struct select_generator_type<scrambled_sobol64> {
            using state_type            = curandStateScrambledSobol64_t;
            using result_type           = unsigned long long;
            using offset_type           = unsigned long long;
            using direction_vector_type = unsigned long long[64];
            using scrambled_const_type  = unsigned long long;
        };


    } // namespace detail
} // namespace curanddx


#endif
