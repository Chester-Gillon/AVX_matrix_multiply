/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_accumulate_nr_c_4_dot_product_length_19 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                            SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                            SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r, left_r0_c17_r, left_r0_c18_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i, left_r0_c17_i, left_r0_c18_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r, left_r1_c17_r, left_r1_c18_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i, left_r1_c17_i, left_r1_c18_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r, left_r2_c17_r, left_r2_c18_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i, left_r2_c17_i, left_r2_c18_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r, left_r3_c5_r, left_r3_c6_r, left_r3_c7_r, left_r3_c8_r, left_r3_c9_r, left_r3_c10_r, left_r3_c11_r, left_r3_c12_r, left_r3_c13_r, left_r3_c14_r, left_r3_c15_r, left_r3_c16_r, left_r3_c17_r, left_r3_c18_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i, left_r3_c5_i, left_r3_c6_i, left_r3_c7_i, left_r3_c8_i, left_r3_c9_i, left_r3_c10_i, left_r3_c11_i, left_r3_c12_i, left_r3_c13_i, left_r3_c14_i, left_r3_c15_i, left_r3_c16_i, left_r3_c17_i, left_r3_c18_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    __m256 output_r3;
    SAL_i32 c_c;
#ifdef IACA_LOAD_LEFT
    IACA_START
#endif

    left_r0_c0_r = _mm256_broadcast_ss (&(A[0] + 0)->real);
    left_r0_c0_i = _mm256_broadcast_ss (&(A[0] + 0)->imag);
    left_r0_c1_r = _mm256_broadcast_ss (&(A[0] + 1)->real);
    left_r0_c1_i = _mm256_broadcast_ss (&(A[0] + 1)->imag);
    left_r0_c2_r = _mm256_broadcast_ss (&(A[0] + 2)->real);
    left_r0_c2_i = _mm256_broadcast_ss (&(A[0] + 2)->imag);
    left_r0_c3_r = _mm256_broadcast_ss (&(A[0] + 3)->real);
    left_r0_c3_i = _mm256_broadcast_ss (&(A[0] + 3)->imag);
    left_r0_c4_r = _mm256_broadcast_ss (&(A[0] + 4)->real);
    left_r0_c4_i = _mm256_broadcast_ss (&(A[0] + 4)->imag);
    left_r0_c5_r = _mm256_broadcast_ss (&(A[0] + 5)->real);
    left_r0_c5_i = _mm256_broadcast_ss (&(A[0] + 5)->imag);
    left_r0_c6_r = _mm256_broadcast_ss (&(A[0] + 6)->real);
    left_r0_c6_i = _mm256_broadcast_ss (&(A[0] + 6)->imag);
    left_r0_c7_r = _mm256_broadcast_ss (&(A[0] + 7)->real);
    left_r0_c7_i = _mm256_broadcast_ss (&(A[0] + 7)->imag);
    left_r0_c8_r = _mm256_broadcast_ss (&(A[0] + 8)->real);
    left_r0_c8_i = _mm256_broadcast_ss (&(A[0] + 8)->imag);
    left_r0_c9_r = _mm256_broadcast_ss (&(A[0] + 9)->real);
    left_r0_c9_i = _mm256_broadcast_ss (&(A[0] + 9)->imag);
    left_r0_c10_r = _mm256_broadcast_ss (&(A[0] + 10)->real);
    left_r0_c10_i = _mm256_broadcast_ss (&(A[0] + 10)->imag);
    left_r0_c11_r = _mm256_broadcast_ss (&(A[0] + 11)->real);
    left_r0_c11_i = _mm256_broadcast_ss (&(A[0] + 11)->imag);
    left_r0_c12_r = _mm256_broadcast_ss (&(A[0] + 12)->real);
    left_r0_c12_i = _mm256_broadcast_ss (&(A[0] + 12)->imag);
    left_r0_c13_r = _mm256_broadcast_ss (&(A[0] + 13)->real);
    left_r0_c13_i = _mm256_broadcast_ss (&(A[0] + 13)->imag);
    left_r0_c14_r = _mm256_broadcast_ss (&(A[0] + 14)->real);
    left_r0_c14_i = _mm256_broadcast_ss (&(A[0] + 14)->imag);
    left_r0_c15_r = _mm256_broadcast_ss (&(A[0] + 15)->real);
    left_r0_c15_i = _mm256_broadcast_ss (&(A[0] + 15)->imag);
    left_r0_c16_r = _mm256_broadcast_ss (&(A[0] + 16)->real);
    left_r0_c16_i = _mm256_broadcast_ss (&(A[0] + 16)->imag);
    left_r0_c17_r = _mm256_broadcast_ss (&(A[0] + 17)->real);
    left_r0_c17_i = _mm256_broadcast_ss (&(A[0] + 17)->imag);
    left_r0_c18_r = _mm256_broadcast_ss (&(A[0] + 18)->real);
    left_r0_c18_i = _mm256_broadcast_ss (&(A[0] + 18)->imag);

    left_r1_c0_r = _mm256_broadcast_ss (&(A[1] + 0)->real);
    left_r1_c0_i = _mm256_broadcast_ss (&(A[1] + 0)->imag);
    left_r1_c1_r = _mm256_broadcast_ss (&(A[1] + 1)->real);
    left_r1_c1_i = _mm256_broadcast_ss (&(A[1] + 1)->imag);
    left_r1_c2_r = _mm256_broadcast_ss (&(A[1] + 2)->real);
    left_r1_c2_i = _mm256_broadcast_ss (&(A[1] + 2)->imag);
    left_r1_c3_r = _mm256_broadcast_ss (&(A[1] + 3)->real);
    left_r1_c3_i = _mm256_broadcast_ss (&(A[1] + 3)->imag);
    left_r1_c4_r = _mm256_broadcast_ss (&(A[1] + 4)->real);
    left_r1_c4_i = _mm256_broadcast_ss (&(A[1] + 4)->imag);
    left_r1_c5_r = _mm256_broadcast_ss (&(A[1] + 5)->real);
    left_r1_c5_i = _mm256_broadcast_ss (&(A[1] + 5)->imag);
    left_r1_c6_r = _mm256_broadcast_ss (&(A[1] + 6)->real);
    left_r1_c6_i = _mm256_broadcast_ss (&(A[1] + 6)->imag);
    left_r1_c7_r = _mm256_broadcast_ss (&(A[1] + 7)->real);
    left_r1_c7_i = _mm256_broadcast_ss (&(A[1] + 7)->imag);
    left_r1_c8_r = _mm256_broadcast_ss (&(A[1] + 8)->real);
    left_r1_c8_i = _mm256_broadcast_ss (&(A[1] + 8)->imag);
    left_r1_c9_r = _mm256_broadcast_ss (&(A[1] + 9)->real);
    left_r1_c9_i = _mm256_broadcast_ss (&(A[1] + 9)->imag);
    left_r1_c10_r = _mm256_broadcast_ss (&(A[1] + 10)->real);
    left_r1_c10_i = _mm256_broadcast_ss (&(A[1] + 10)->imag);
    left_r1_c11_r = _mm256_broadcast_ss (&(A[1] + 11)->real);
    left_r1_c11_i = _mm256_broadcast_ss (&(A[1] + 11)->imag);
    left_r1_c12_r = _mm256_broadcast_ss (&(A[1] + 12)->real);
    left_r1_c12_i = _mm256_broadcast_ss (&(A[1] + 12)->imag);
    left_r1_c13_r = _mm256_broadcast_ss (&(A[1] + 13)->real);
    left_r1_c13_i = _mm256_broadcast_ss (&(A[1] + 13)->imag);
    left_r1_c14_r = _mm256_broadcast_ss (&(A[1] + 14)->real);
    left_r1_c14_i = _mm256_broadcast_ss (&(A[1] + 14)->imag);
    left_r1_c15_r = _mm256_broadcast_ss (&(A[1] + 15)->real);
    left_r1_c15_i = _mm256_broadcast_ss (&(A[1] + 15)->imag);
    left_r1_c16_r = _mm256_broadcast_ss (&(A[1] + 16)->real);
    left_r1_c16_i = _mm256_broadcast_ss (&(A[1] + 16)->imag);
    left_r1_c17_r = _mm256_broadcast_ss (&(A[1] + 17)->real);
    left_r1_c17_i = _mm256_broadcast_ss (&(A[1] + 17)->imag);
    left_r1_c18_r = _mm256_broadcast_ss (&(A[1] + 18)->real);
    left_r1_c18_i = _mm256_broadcast_ss (&(A[1] + 18)->imag);

    left_r2_c0_r = _mm256_broadcast_ss (&(A[2] + 0)->real);
    left_r2_c0_i = _mm256_broadcast_ss (&(A[2] + 0)->imag);
    left_r2_c1_r = _mm256_broadcast_ss (&(A[2] + 1)->real);
    left_r2_c1_i = _mm256_broadcast_ss (&(A[2] + 1)->imag);
    left_r2_c2_r = _mm256_broadcast_ss (&(A[2] + 2)->real);
    left_r2_c2_i = _mm256_broadcast_ss (&(A[2] + 2)->imag);
    left_r2_c3_r = _mm256_broadcast_ss (&(A[2] + 3)->real);
    left_r2_c3_i = _mm256_broadcast_ss (&(A[2] + 3)->imag);
    left_r2_c4_r = _mm256_broadcast_ss (&(A[2] + 4)->real);
    left_r2_c4_i = _mm256_broadcast_ss (&(A[2] + 4)->imag);
    left_r2_c5_r = _mm256_broadcast_ss (&(A[2] + 5)->real);
    left_r2_c5_i = _mm256_broadcast_ss (&(A[2] + 5)->imag);
    left_r2_c6_r = _mm256_broadcast_ss (&(A[2] + 6)->real);
    left_r2_c6_i = _mm256_broadcast_ss (&(A[2] + 6)->imag);
    left_r2_c7_r = _mm256_broadcast_ss (&(A[2] + 7)->real);
    left_r2_c7_i = _mm256_broadcast_ss (&(A[2] + 7)->imag);
    left_r2_c8_r = _mm256_broadcast_ss (&(A[2] + 8)->real);
    left_r2_c8_i = _mm256_broadcast_ss (&(A[2] + 8)->imag);
    left_r2_c9_r = _mm256_broadcast_ss (&(A[2] + 9)->real);
    left_r2_c9_i = _mm256_broadcast_ss (&(A[2] + 9)->imag);
    left_r2_c10_r = _mm256_broadcast_ss (&(A[2] + 10)->real);
    left_r2_c10_i = _mm256_broadcast_ss (&(A[2] + 10)->imag);
    left_r2_c11_r = _mm256_broadcast_ss (&(A[2] + 11)->real);
    left_r2_c11_i = _mm256_broadcast_ss (&(A[2] + 11)->imag);
    left_r2_c12_r = _mm256_broadcast_ss (&(A[2] + 12)->real);
    left_r2_c12_i = _mm256_broadcast_ss (&(A[2] + 12)->imag);
    left_r2_c13_r = _mm256_broadcast_ss (&(A[2] + 13)->real);
    left_r2_c13_i = _mm256_broadcast_ss (&(A[2] + 13)->imag);
    left_r2_c14_r = _mm256_broadcast_ss (&(A[2] + 14)->real);
    left_r2_c14_i = _mm256_broadcast_ss (&(A[2] + 14)->imag);
    left_r2_c15_r = _mm256_broadcast_ss (&(A[2] + 15)->real);
    left_r2_c15_i = _mm256_broadcast_ss (&(A[2] + 15)->imag);
    left_r2_c16_r = _mm256_broadcast_ss (&(A[2] + 16)->real);
    left_r2_c16_i = _mm256_broadcast_ss (&(A[2] + 16)->imag);
    left_r2_c17_r = _mm256_broadcast_ss (&(A[2] + 17)->real);
    left_r2_c17_i = _mm256_broadcast_ss (&(A[2] + 17)->imag);
    left_r2_c18_r = _mm256_broadcast_ss (&(A[2] + 18)->real);
    left_r2_c18_i = _mm256_broadcast_ss (&(A[2] + 18)->imag);

    left_r3_c0_r = _mm256_broadcast_ss (&(A[3] + 0)->real);
    left_r3_c0_i = _mm256_broadcast_ss (&(A[3] + 0)->imag);
    left_r3_c1_r = _mm256_broadcast_ss (&(A[3] + 1)->real);
    left_r3_c1_i = _mm256_broadcast_ss (&(A[3] + 1)->imag);
    left_r3_c2_r = _mm256_broadcast_ss (&(A[3] + 2)->real);
    left_r3_c2_i = _mm256_broadcast_ss (&(A[3] + 2)->imag);
    left_r3_c3_r = _mm256_broadcast_ss (&(A[3] + 3)->real);
    left_r3_c3_i = _mm256_broadcast_ss (&(A[3] + 3)->imag);
    left_r3_c4_r = _mm256_broadcast_ss (&(A[3] + 4)->real);
    left_r3_c4_i = _mm256_broadcast_ss (&(A[3] + 4)->imag);
    left_r3_c5_r = _mm256_broadcast_ss (&(A[3] + 5)->real);
    left_r3_c5_i = _mm256_broadcast_ss (&(A[3] + 5)->imag);
    left_r3_c6_r = _mm256_broadcast_ss (&(A[3] + 6)->real);
    left_r3_c6_i = _mm256_broadcast_ss (&(A[3] + 6)->imag);
    left_r3_c7_r = _mm256_broadcast_ss (&(A[3] + 7)->real);
    left_r3_c7_i = _mm256_broadcast_ss (&(A[3] + 7)->imag);
    left_r3_c8_r = _mm256_broadcast_ss (&(A[3] + 8)->real);
    left_r3_c8_i = _mm256_broadcast_ss (&(A[3] + 8)->imag);
    left_r3_c9_r = _mm256_broadcast_ss (&(A[3] + 9)->real);
    left_r3_c9_i = _mm256_broadcast_ss (&(A[3] + 9)->imag);
    left_r3_c10_r = _mm256_broadcast_ss (&(A[3] + 10)->real);
    left_r3_c10_i = _mm256_broadcast_ss (&(A[3] + 10)->imag);
    left_r3_c11_r = _mm256_broadcast_ss (&(A[3] + 11)->real);
    left_r3_c11_i = _mm256_broadcast_ss (&(A[3] + 11)->imag);
    left_r3_c12_r = _mm256_broadcast_ss (&(A[3] + 12)->real);
    left_r3_c12_i = _mm256_broadcast_ss (&(A[3] + 12)->imag);
    left_r3_c13_r = _mm256_broadcast_ss (&(A[3] + 13)->real);
    left_r3_c13_i = _mm256_broadcast_ss (&(A[3] + 13)->imag);
    left_r3_c14_r = _mm256_broadcast_ss (&(A[3] + 14)->real);
    left_r3_c14_i = _mm256_broadcast_ss (&(A[3] + 14)->imag);
    left_r3_c15_r = _mm256_broadcast_ss (&(A[3] + 15)->real);
    left_r3_c15_i = _mm256_broadcast_ss (&(A[3] + 15)->imag);
    left_r3_c16_r = _mm256_broadcast_ss (&(A[3] + 16)->real);
    left_r3_c16_i = _mm256_broadcast_ss (&(A[3] + 16)->imag);
    left_r3_c17_r = _mm256_broadcast_ss (&(A[3] + 17)->real);
    left_r3_c17_i = _mm256_broadcast_ss (&(A[3] + 17)->imag);
    left_r3_c18_r = _mm256_broadcast_ss (&(A[3] + 18)->real);
    left_r3_c18_i = _mm256_broadcast_ss (&(A[3] + 18)->imag);
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
        right_r_i = _mm256_load_ps (&(B[0] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r2_c0_i));
        output_r3 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r3_c0_i));
        right_r_i = _mm256_load_ps (&(B[1] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c1_i)));
        right_r_i = _mm256_load_ps (&(B[2] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c2_i)));
        right_r_i = _mm256_load_ps (&(B[3] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c3_i)));
        right_r_i = _mm256_load_ps (&(B[4] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c4_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c4_i)));
        right_r_i = _mm256_load_ps (&(B[5] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c5_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c5_i)));
        right_r_i = _mm256_load_ps (&(B[6] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c6_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c6_i)));
        right_r_i = _mm256_load_ps (&(B[7] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c7_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c7_i)));
        right_r_i = _mm256_load_ps (&(B[8] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c8_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c8_i)));
        right_r_i = _mm256_load_ps (&(B[9] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c9_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c9_i)));
        right_r_i = _mm256_load_ps (&(B[10] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c10_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c10_i)));
        right_r_i = _mm256_load_ps (&(B[11] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c11_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c11_i)));
        right_r_i = _mm256_load_ps (&(B[12] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c12_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c12_i)));
        right_r_i = _mm256_load_ps (&(B[13] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c13_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c13_i)));
        right_r_i = _mm256_load_ps (&(B[14] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c14_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c14_i)));
        right_r_i = _mm256_load_ps (&(B[15] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c15_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c15_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c15_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c15_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c15_i)));
        right_r_i = _mm256_load_ps (&(B[16] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c16_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c16_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c16_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c16_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c16_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c16_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c16_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c16_i)));
        right_r_i = _mm256_load_ps (&(B[17] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c17_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c17_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c17_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c17_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c17_r),
                                                                _mm256_mul_ps (right_i_r, left_r2_c17_i)));
        output_r3 = _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c17_r),
                                                                _mm256_mul_ps (right_i_r, left_r3_c17_i)));
        right_r_i = _mm256_load_ps (&(B[18] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c18_r),
                                                                                          _mm256_mul_ps (right_i_r, left_r0_c18_i))));
        _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c18_r),
                                                                                          _mm256_mul_ps (right_i_r, left_r1_c18_i))));
        _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c18_r),
                                                                                          _mm256_mul_ps (right_i_r, left_r2_c18_i))));
        _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (output_r3, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c18_r),
                                                                                          _mm256_mul_ps (right_i_r, left_r3_c18_i))));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

