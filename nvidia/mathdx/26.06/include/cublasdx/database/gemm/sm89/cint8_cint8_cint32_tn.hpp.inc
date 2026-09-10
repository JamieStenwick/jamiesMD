
template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<256, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 8>,
    generated_config_impl<32, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 16, 2>
> {};

template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<512, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<256, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 1, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 48, 48, 48, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 64, 64, 64, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 80, 80, 80, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<int8_t>, cutlass::complex<int8_t>, cutlass::complex<int32_t>, 91, 91, 91, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM80_8x8x16_CS32CS8CS8CS32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_16x8x32_CS32CS8CS8CS32_TN_CUBLASDX, 1, 1>
> {};
