// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_LOG_NORMAL_HPP
#define CURANDDX_DETAIL_RAND_LOG_NORMAL_HPP

#include "curanddx/distributions/rand_normal.hpp"

namespace curanddx {

    template<typename FPType = float, normal_method Method = icdf>
    struct log_normal {

        static_assert(COMMONDX_STL_NAMESPACE::is_floating_point_v<FPType>, "log normal supports floating point types only");

        using result_type  = FPType;
        using result2_type = typename detail::make_vector<FPType, 2>::type;
        using result4_type = typename detail::make_vector<FPType, 4>::type;

        // ctor
        CURANDDX_QUALIFIERS log_normal(FPType mean = 0, FPType stddev = 1): m_mean(mean), m_stddev(stddev) {};

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
                          "pack size should be 1 to use the log normal function generate()");

            normal<FPType, Method> my_normal(0, 1);
            return exp(m_mean + m_stddev * my_normal.generate(rng));
        }

        // Only for Box-Muller used only by non-philox generator, Or Box-Muller used by Philox for double only
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate2(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(Method == box_muller &&
                                                    (!RNG::is_generator_philox ||
                                                     (RNG::is_generator_philox && COMMONDX_STL_NAMESPACE::is_same_v<FPType, double>))),
                                                   result2_type> {
            static_assert(min_pack_size<generator_of_v<RNG>> <= 2 && max_pack_size<generator_of_v<RNG>> >= 2,
                          "pack size should be 1 to use the log normal function generate2()");


            normal<FPType, Method> my_normal(0, 1);
            const auto [a, b] = my_normal.generate2(rng);

            return {exp(m_mean + m_stddev * a), exp(m_mean + m_stddev * b)};
        }

        // Only for normal4 or normal4_double for Philox, using both ICDF and Box-Muller
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate4(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<(RNG::is_generator_philox), result4_type> {

            static_assert(min_pack_size<generator_of_v<RNG>> <= 4 && max_pack_size<generator_of_v<RNG>> >= 4,
                          "pack size should be 1 to use the log normal function generate4()");

            normal<FPType, Method> my_normal(0, 1);
            const auto [a, b, c, d] = my_normal.generate4(rng);

            return {exp(m_mean + m_stddev * a),
                    exp(m_mean + m_stddev * b),
                    exp(m_mean + m_stddev * c),
                    exp(m_mean + m_stddev * d)};
        }


    private:
        FPType m_mean, m_stddev;
    };


} // namespace curanddx

#endif // CURANDDX_DETAIL_RAND_LOG_NORMAL_HPP
