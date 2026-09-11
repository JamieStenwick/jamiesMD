// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CURANDDX_DETAIL_RAND_POISSON_HPP
#define CURANDDX_DETAIL_RAND_POISSON_HPP

#include "curanddx/detail/common.hpp"
#include "curanddx/distributions/rand_uniform.hpp"
#include "curanddx/distributions/rand_normal.hpp"

namespace curanddx {

    struct poisson {

        using result_type = unsigned int;

        // ctor
        CURANDDX_QUALIFIERS poisson(double lambda = 1): m_lambda(lambda) {};

        // get set
        CURANDDX_QUALIFIERS double lambda() const { return m_lambda; };

        // pack size variables
        template<generator GEN>
        static constexpr unsigned int min_pack_size = 1;
        template<generator GEN>
        static constexpr unsigned int max_pack_size = 1;

        // Poisson does not support Philox
        template<class RNG>
        CURANDDX_QUALIFIERS auto generate(RNG& rng)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<!RNG::is_generator_philox, result_type> {

            result_type result;
            if constexpr (generator_of_v<RNG> == mtgp32 || RNG::is_generator_sobol) {
                result = ::_curand_poisson<RNG::bitgenerator_result_type>(rng.generate(), m_lambda);
            } else { // For xorwow, mrg, and pcg generators
                if (m_lambda < 64) {
                    result = rand_poisson_knuth(rng);
                } else if (m_lambda > 4000) {
                    normal<double, icdf> my_normal(0, 1);
                    result =
                        (unsigned int)((sqrt(m_lambda) * my_normal.generate(rng)) + m_lambda + 0.5); //Round to nearest
                } else {
                    result = rand_poisson_gammainc(rng);
                }
            }
            return result;
        }


    private:
        double m_lambda;

        // Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming, Volume 2
        template<class RNG>
        CURANDDX_QUALIFIERS unsigned int rand_poisson_knuth(RNG& rng) {
            float          p = expf(float(m_lambda));
            uniform<float> my_uniform(0, 1);
            unsigned int   k = 0;
            do {
                k++;
                p *= my_uniform.generate(rng);
            } while (p > 1.0);
            return k - 1;
        }

        // Rejection Method for Poisson distribution based on gammainc approximation
        template<class RNG>
        CURANDDX_QUALIFIERS unsigned int rand_poisson_gammainc(RNG& rng) {
            float lambda = (float)m_lambda;

            uniform<float> my_uniform(0, 1);

            float y, x, t, z, v;
            float logl = __cr_log(lambda);
            while (true) {
                y = my_uniform.generate(rng);
                x = __cr_pgammaincinv(lambda, y);
                x = floorf(x);
                z = my_uniform.generate(rng);
                v = (__cr_pgammainc(lambda, x + 1.0f) - __cr_pgammainc(lambda, x)) * 1.3f;
                z = z * v;
                t = (float)__cr_exp(-lambda + x * logl - (float)__cr_lgamma_integer((int)(1.0f + x)));
                if ((z < t) && (v >= 1e-20))
                    break;
            }
            return (unsigned int)x;
        }
    };

} // namespace curanddx


#endif // CURANDDX_DETAIL_RAND_POISSON_HPP
