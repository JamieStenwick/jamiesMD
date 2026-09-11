// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_UNIFORM_HPP
#define CURANDDX_DETAIL_RAND_UNIFORM_HPP

#include "curanddx/detail/common.hpp"

namespace curanddx {
    namespace detail {

        CURANDDX_QUALIFIERS float rand_uniform(unsigned int x) {
            return x * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
        }
        CURANDDX_QUALIFIERS float rand_uniform(unsigned long long x) {
            unsigned int t;
            t = (unsigned int)(x >> 32);
            return t * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
        }
        CURANDDX_QUALIFIERS float4 rand_uniform4(uint4 x) {
            float4 y;
            y.x = x.x * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
            y.y = x.y * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
            y.z = x.z * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
            y.w = x.w * rand_2pow32_inv<float> + (rand_2pow32_inv<float> / 2.0f);
            return y;
        }
        CURANDDX_QUALIFIERS double rand_uniform_double_hq(unsigned int x, unsigned int y) {
            unsigned long long z = (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
            return z * rand_2pow53_inv_double + (rand_2pow53_inv_double / 2.0);
        }
        // The following functions as used for sobol
        CURANDDX_QUALIFIERS double rand_uniform_double_sobol(unsigned int x) {
            return double(x) * detail::rand_2pow32_inv<double> + detail::rand_2pow32_inv<double>;
        }
        CURANDDX_QUALIFIERS double rand_uniform_double_sobol(unsigned long long x) {
            return double(x >> 11) * detail::rand_2pow53_inv_double + (detail::rand_2pow53_inv_double / 2.0);
        }

        template<generator GEN, typename FPType = float>
        struct uniform_min_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<>
        struct uniform_min_pack_size<philox4_32, float> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct uniform_min_pack_size<philox4_32, double> {
            static constexpr unsigned int value = 2;
        };
        template<generator GEN, typename FPType = float>
        struct uniform_max_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<>
        struct uniform_max_pack_size<philox4_32, float> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct uniform_max_pack_size<philox4_32, double> {
            static constexpr unsigned int value = 4;
        };

    } // namespace detail


    template<typename FPType = float>
    struct uniform {
        static_assert(COMMONDX_STL_NAMESPACE::is_floating_point_v<FPType>, "uniform supports floating point types only");

        using result_type  = FPType;
        using result2_type = typename detail::make_vector<FPType, 2>::type;
        using result4_type = typename detail::make_vector<FPType, 4>::type;

        // ctor
        CURANDDX_QUALIFIERS uniform(FPType min = 0, FPType max = 1): m_min(min), m_range(max - min) {};

        // get set
        CURANDDX_QUALIFIERS FPType min() const { return m_min; };
        CURANDDX_QUALIFIERS FPType max() const { return m_min + m_range; };

        // pack size variables
        template<generator GEN>
        static constexpr unsigned int min_pack_size = detail::uniform_min_pack_size<GEN, FPType>::value;
        template<generator GEN>
        static constexpr unsigned int max_pack_size = detail::uniform_max_pack_size<GEN, FPType>::value;


        // generate functions
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<!RNG::is_generator_philox, result_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 1 && max_pack_size<generator_of_v<RNG>> >= 1,
                          "pack size should be 1 to use the uniform function generate()");

            result_type result;
            // MRG generator can directly generates double number
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                if constexpr (generator_of_v<RNG> == mrg32k3a) {
                    result = ::curand_uniform(&(rng.get_generator()())) * m_range + m_min;
                } else {
                    result = detail::rand_uniform(rng.generate()) * m_range + m_min;
                }
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                if constexpr (RNG::is_generator_sobol || generator_of_v<RNG> == mtgp32) {
                    result = detail::rand_uniform_double_sobol(rng.generate()) * m_range + m_min;
#ifdef CURANDDX_MRG_DOUBLE_DISTRIBUTION_CURAND_COMPATIBLE
                } else if constexpr (generator_of_v<RNG> == mrg32k3a) {
                    result = ::curand_uniform_double(&(rng.get_generator()())) * m_range + m_min;
#endif
                } else {
                    result = detail::rand_uniform_double_hq(rng.generate(), rng.generate()) * m_range + m_min;
                }
            }
            return result;
        }

        // Only for uniform_double2 for Philox
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate2(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(COMMONDX_STL_NAMESPACE::is_same_v<FPType, double> && RNG::is_generator_philox),
                                                   result2_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 2 && max_pack_size<generator_of_v<RNG>> >= 2,
                          "pack size should be 1 to use the uniform function generate2()");

            const auto [a, b, c, d] = rng.generate4();
            double2 result;
            result.x = detail::rand_uniform_double_hq(a, b) * m_range + m_min;
            result.y = detail::rand_uniform_double_hq(c, d) * m_range + m_min;

            return result;
        }

        // Only for uniform4 or uniform4_double for Philox
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate4(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_generator_philox), result4_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 4 && max_pack_size<generator_of_v<RNG>> >= 4,
                          "min pack size of the generator should be 4 to use the uniform generate4()");

            result4_type result;
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                result = detail::rand_uniform4(rng.generate4());
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                const auto [a, b, c, d] = rng.generate4();
                const auto [e, f, g, h] = rng.generate4();
                result.x                 = detail::rand_uniform_double_hq(a, b);
                result.y                 = detail::rand_uniform_double_hq(c, d);
                result.z                 = detail::rand_uniform_double_hq(e, f);
                result.w                 = detail::rand_uniform_double_hq(g, h);
            }
            return {result.x * m_range + m_min,
                    result.y * m_range + m_min,
                    result.z * m_range + m_min,
                    result.w * m_range + m_min};
        }


    private:
        FPType m_min, m_range;
    };

} // namespace curanddx


#endif // CURANDDX_DETAIL_RAND_UNIFORM_HPP
