
template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 8>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 16>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 32>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 16>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 48, 48, 48, SM< 890 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 64, 64, 64, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 80, 80, 80, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 96, 96, 96, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 112, 112, 112, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 128, 128, 128, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 144, 144, 144, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 16, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 160, 160, 160, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 176, 176, 176, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 192, 192, 192, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 208, 208, 208, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 16, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 224, 224, 224, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, half_t, 225, 225, 225, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x16_F16E5M2E5M2F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F16E5M2E5M2F16_TN_CUBLASDX, 1, 1>
> {};
