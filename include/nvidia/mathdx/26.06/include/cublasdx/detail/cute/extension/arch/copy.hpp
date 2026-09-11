#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COPY_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COPY_HPP

namespace cublasdx {
    namespace detail {
        //
        // Copy policy automatically selecting between
        // UniversalCopy and cp.async , based on type and memory space.
        // Do not bypass L1 cache.
        //
        struct auto_copy_async_cache_always_cublasdx {
        };

        //
        // Copy policy automatically selecting between
        // UniversalCopy and cp.async , based on type and memory space.
        // Attempt to bypass L1 cache on proper alignment (16 bytes required).
        struct auto_copy_async_cache_global_cublasdx {
        };
    } // namespace detail
} // namespace cublasdx
#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_ARCH_COPY_HPP
