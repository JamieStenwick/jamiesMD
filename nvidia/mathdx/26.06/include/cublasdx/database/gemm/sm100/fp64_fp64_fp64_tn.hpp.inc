
template<>
struct generated_config<double, double, double, 16, 16, 16, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>
> {};

template<>
struct generated_config<double, double, double, 32, 32, 32, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 1>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 48, 48, 48, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 1>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 8>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 64, 64, 64, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 80, 80, 80, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 8>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 96, 96, 96, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 8>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 2>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 112, 112, 112, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 8>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 120, 120, 120, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 8>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 16>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 2>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};
