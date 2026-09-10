// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_UNIFORM_BITS_HPP
#define CURANDDX_DETAIL_RAND_UNIFORM_BITS_HPP

#include "curanddx/detail/common.hpp"

namespace curanddx {

    namespace detail {
        template<generator GEN>
        struct uniform_bits_min_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<>
        struct uniform_bits_min_pack_size<philox4_32> {
            static constexpr unsigned int value = 4;
        };
        template<generator GEN>
        struct uniform_bits_max_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<>
        struct uniform_bits_max_pack_size<philox4_32> {
            static constexpr unsigned int value = 4;
        };
    } // namespace detail

    template<typename UIntType = unsigned int>
    struct uniform_bits {
        static_assert(COMMONDX_STL_NAMESPACE::is_same_v<UIntType, unsigned int> || COMMONDX_STL_NAMESPACE::is_same_v<UIntType, unsigned long long>,
                      "uniform_bits supports only unsigned int or unsigned long long");

        using result_type  = UIntType;
        using result4_type = typename detail::make_vector<UIntType, 4>::type;

        // ctor
        CURANDDX_QUALIFIERS uniform_bits() {};

        // pack size variables
        template<generator GEN>
        static constexpr unsigned int min_pack_size = detail::uniform_bits_min_pack_size<GEN>::value;
        template<generator GEN>
        static constexpr unsigned int max_pack_size = detail::uniform_bits_max_pack_size<GEN>::value;

        // generate one number, either unsigned int or unsigned long long, depending on either 32-bit or 64-bit generator
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate(RNG& rng) -> COMMONDX_STL_NAMESPACE::enable_if_t<
            (RNG::is_generator_32bits && COMMONDX_STL_NAMESPACE::is_same_v<UIntType, unsigned int>) ||
                (RNG::is_generator_64bits && COMMONDX_STL_NAMESPACE::is_same_v<UIntType, unsigned long long>),
            result_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 1 && max_pack_size<generator_of_v<RNG>> >= 1,
                          "pack size should be 1 to use the uniform bits function generate()");


            return rng.generate();
        }

        // Only for Philox
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate4(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_generator_philox && COMMONDX_STL_NAMESPACE::is_same_v<UIntType, unsigned int>),
                                                   result4_type> {
            static_assert(min_pack_size<generator_of_v<RNG>> <= 4 && max_pack_size<generator_of_v<RNG>> >= 4,
                          "pack size should be 1 to use the uniform bits function generate()");
            return rng.generate4();
        }
    };

} // namespace curanddx


#endif // CURANDDX_DETAIL_RAND_UNIFORM_BITS_HPP
