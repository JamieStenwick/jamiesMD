
template<>
struct generated_config<int8_t, int8_t, int32_t, 16, 16, 16, SM< 750 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 32, 32, 32, SM< 750 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 4, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 48, 48, 48, SM< 750 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 64, 64, 64, SM< 750 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 4, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 8, 64>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 80, 80, 80, SM< 750 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 96, 96, 96, SM< 750 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 4, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 8, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<int8_t, int8_t, int32_t, 104, 104, 104, SM< 750 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 8, 64>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};
