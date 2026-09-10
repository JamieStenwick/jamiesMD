// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_NORMAL_HPP
#define CURANDDX_DETAIL_RAND_NORMAL_HPP

#include "curanddx/detail/common.hpp"
#include <curand_normal.h>

namespace curanddx {

    enum class normal_method
    {
        icdf,
        box_muller
    };
    inline constexpr auto icdf       = normal_method::icdf;
    inline constexpr auto box_muller = normal_method::box_muller;


    namespace detail {
        template<generator GEN, typename FPType, normal_method Method = icdf>
        struct normal_min_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<generator GEN, typename FPType>
        struct normal_min_pack_size<GEN, FPType, box_muller> {
            static constexpr unsigned int value = 2;
        };
        template<>
        struct normal_min_pack_size<philox4_32, float, box_muller> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct normal_min_pack_size<philox4_32, float, icdf> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct normal_min_pack_size<philox4_32, double, box_muller> {
            static constexpr unsigned int value = 2; // note that cuRAND use curand_double4
        };
        template<>
        struct normal_min_pack_size<philox4_32, double, icdf> {
            static constexpr unsigned int value = 4;
        };

        template<generator GEN, typename FPType, normal_method Method = icdf>
        struct normal_max_pack_size {
            static constexpr unsigned int value = 1;
        };
        template<generator GEN, typename FPType>
        struct normal_max_pack_size<GEN, FPType, box_muller> {
            static constexpr unsigned int value = 2;
        };
        template<>
        struct normal_max_pack_size<philox4_32, float, box_muller> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct normal_max_pack_size<philox4_32, float, icdf> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct normal_max_pack_size<philox4_32, double, box_muller> {
            static constexpr unsigned int value = 4;
        };
        template<>
        struct normal_max_pack_size<philox4_32, double, icdf> {
            static constexpr unsigned int value = 4;
        };


    } // namespace detail


    template<typename FPType = float, normal_method Method = icdf>
    struct normal {

        static_assert(COMMONDX_STL_NAMESPACE::is_floating_point_v<FPType>, "normal supports floating point types only");

        using result_type  = FPType;
        using result2_type = typename detail::make_vector<FPType, 2>::type;
        using result4_type = typename detail::make_vector<FPType, 4>::type;

        // ctor
        CURANDDX_QUALIFIERS normal(FPType mean = 0, FPType stddev = 1): m_mean(mean), m_stddev(stddev) {};

        // get set
        CURANDDX_QUALIFIERS FPType mean() const { return m_mean; };
        CURANDDX_QUALIFIERS FPType stddev() const { return m_stddev; };

        // pack size variables
        template<generator GEN>
        static constexpr unsigned int min_pack_size = detail::normal_min_pack_size<GEN, FPType, Method>::value;
        template<generator GEN>
        static constexpr unsigned int max_pack_size = detail::normal_max_pack_size<GEN, FPType, Method>::value;

        // generate one number. Not allowed for Box-Muller. Not allowed for Philox generator
        // ICDF generates can generate double using either unsigned int or unsigned long long
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(!RNG::is_generator_philox && Method != box_muller), result_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 1 && max_pack_size<generator_of_v<RNG>> >= 1,
                          "pack size should be 1 to use the normal function generate()");

            result_type result;
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                result = ::_curand_normal_icdf(rng.generate()) * m_stddev + m_mean;
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                result = ::_curand_normal_icdf_double(rng.generate()) * m_stddev + m_mean;
            }
            return result;
        }

        // Only for Box-Muller used only by non-philox generator, Or Box-Muller used by Philox for double only
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate2(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(Method == box_muller &&
                                                    (!RNG::is_generator_philox ||
                                                     (RNG::is_generator_philox && COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>))),
                                                   result2_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 2 && max_pack_size<generator_of_v<RNG>> >= 2,
                          "pack size should be 1 to use the normal function generate2()");

            result2_type result;
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                if constexpr (generator_of_v<RNG> == mrg32k3a) {
                    result = ::curand_box_muller_mrg(&(rng.get_generator()()));
                } else {
                    result = ::_curand_box_muller(rng.generate(), rng.generate());
                }
            } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                if constexpr (RNG::is_generator_philox) { // Philox double
                    const auto [a, b, c, d] = rng.generate4();
                    result                  = ::_curand_box_muller_double(a, b, c, d);
#ifdef CURANDDX_MRG_DOUBLE_DISTRIBUTION_CURAND_COMPATIBLE
                } else if constexpr (generator_of_v<RNG> == mrg32k3a) {
                    result = ::curand_box_muller_mrg_double(&(rng.get_generator()()));
#endif
                } else {
                    result =
                        ::_curand_box_muller_double(rng.generate(), rng.generate(), rng.generate(), rng.generate());
                }
            }

            return {m_stddev * result.x + m_mean, m_stddev * result.y + m_mean};
        }

        // Only for normal4 or normal4_double for Philox, using both ICDF and Box-Muller
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate4(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_generator_philox), result4_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 4 && max_pack_size<generator_of_v<RNG>> >= 4,
                          "min pack size of the generator should be 4 to use the normal generate4()");

            result4_type result;
            if constexpr (Method == icdf) {
                if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                    const auto [a, b, c, d] = rng.generate4();
                    result.x                = ::_curand_normal_icdf(a) * m_stddev + m_mean;
                    result.y                = ::_curand_normal_icdf(b) * m_stddev + m_mean;
                    result.z                = ::_curand_normal_icdf(c) * m_stddev + m_mean;
                    result.w                = ::_curand_normal_icdf(d) * m_stddev + m_mean;
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                    const auto [a, b, c, d] = rng.generate4();
                    result.x                = ::_curand_normal_icdf_double(a) * m_stddev + m_mean;
                    result.y                = ::_curand_normal_icdf_double(b) * m_stddev + m_mean;
                    result.z                = ::_curand_normal_icdf_double(c) * m_stddev + m_mean;
                    result.w                = ::_curand_normal_icdf_double(d) * m_stddev + m_mean;
                }
            } else { // box-muller method
                if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, float>) {
                    const auto [a, b, c, d] = rng.generate4();
                    float2 result1          = ::_curand_box_muller(a, b);
                    float2 result2          = ::_curand_box_muller(c, d);

                    result = {m_stddev * result1.x + m_mean,
                              m_stddev * result1.y + m_mean,
                              m_stddev * result2.x + m_mean,
                              m_stddev * result2.y + m_mean};
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>) {
                    const auto [a, b, c, d] = rng.generate4();
                    double2 result1         = ::_curand_box_muller_double(a, b, c, d);
                    const auto [e, f, g, h] = rng.generate4();
                    double2 result2         = ::_curand_box_muller_double(e, f, g, h);

                    result = {m_stddev * result1.x + m_mean,
                              m_stddev * result1.y + m_mean,
                              m_stddev * result2.x + m_mean,
                              m_stddev * result2.y + m_mean};
                }
            }
            return result;
        }

    private:
        FPType m_mean, m_stddev;
    };


} // namespace curanddx

#endif // CURANDDX_DETAIL_RAND_NORMAL_HPP
