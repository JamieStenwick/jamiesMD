
template<>
struct generated_config<float, float, float, 16, 16, 16, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>
> {};

template<>
struct generated_config<float, float, float, 32, 32, 32, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 8>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 4>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 32>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>
> {};

template<>
struct generated_config<float, float, float, 48, 48, 48, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 4, 8>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 64, 16>
> {};

template<>
struct generated_config<float, float, float, 64, 64, 64, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 8>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 8>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 80, 80, 80, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 64, 16>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 4, 8>
> {};

template<>
struct generated_config<float, float, float, 96, 96, 96, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 32>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 112, 112, 112, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 64, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 4, 8>
> {};

template<>
struct generated_config<float, float, float, 128, 128, 128, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 64, 16>,
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 144, 144, 144, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 32>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>,
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 8>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 160, 160, 160, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 32>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 32>,
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 8>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 2>
> {};

template<>
struct generated_config<float, float, float, 170, 170, 170, SM< 1000 >> : generated_config_list<
    
    generated_config_impl<512, mma_atom::SM100_2x1x1_F32F32F32F32, 32, 16>,
    generated_config_impl<1024, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 64>,
    generated_config_impl<256, mma_atom::SM100_2x1x1_F32F32F32F32, 16, 16>,
    generated_config_impl<128, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 16>,
    generated_config_impl<64, mma_atom::SM100_2x1x1_F32F32F32F32, 4, 16>,
    generated_config_impl<32, mma_atom::SM100_2x1x1_F32F32F32F32, 8, 4>
> {};
