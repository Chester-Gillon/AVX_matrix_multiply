/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_stripmine_nr_c_19_dot_product_length_2 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                           SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                           SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    const SAL_i32 nr_c = 19;
    /*const SAL_i32 dot_product_length = 2;*/
    const SAL_i32 L1_cache_size = 32768;
    const SAL_i32 strip_size_samples = (L1_cache_size / sizeof(__m256) / (nr_c + 2)) / 4 * 4;
    __m256 left_r0_c0_r, left_r0_c1_r;
    __m256 left_r0_c0_i, left_r0_c1_i;
    __m256 left_r1_c0_r, left_r1_c1_r;
    __m256 left_r1_c0_i, left_r1_c1_i;
    __m256 left_r2_c0_r, left_r2_c1_r;
    __m256 left_r2_c0_i, left_r2_c1_i;
    __m256 left_r3_c0_r, left_r3_c1_r;
    __m256 left_r3_c0_i, left_r3_c1_i;
    __m256 left_r4_c0_r, left_r4_c1_r;
    __m256 left_r4_c0_i, left_r4_c1_i;
    __m256 left_r5_c0_r, left_r5_c1_r;
    __m256 left_r5_c0_i, left_r5_c1_i;
    __m256 left_r6_c0_r, left_r6_c1_r;
    __m256 left_r6_c0_i, left_r6_c1_i;
    __m256 left_r7_c0_r, left_r7_c1_r;
    __m256 left_r7_c0_i, left_r7_c1_i;
    __m256 left_r8_c0_r, left_r8_c1_r;
    __m256 left_r8_c0_i, left_r8_c1_i;
    __m256 left_r9_c0_r, left_r9_c1_r;
    __m256 left_r9_c0_i, left_r9_c1_i;
    __m256 left_r10_c0_r, left_r10_c1_r;
    __m256 left_r10_c0_i, left_r10_c1_i;
    __m256 left_r11_c0_r, left_r11_c1_r;
    __m256 left_r11_c0_i, left_r11_c1_i;
    __m256 left_r12_c0_r, left_r12_c1_r;
    __m256 left_r12_c0_i, left_r12_c1_i;
    __m256 left_r13_c0_r, left_r13_c1_r;
    __m256 left_r13_c0_i, left_r13_c1_i;
    __m256 left_r14_c0_r, left_r14_c1_r;
    __m256 left_r14_c0_i, left_r14_c1_i;
    __m256 left_r15_c0_r, left_r15_c1_r;
    __m256 left_r15_c0_i, left_r15_c1_i;
    __m256 left_r16_c0_r, left_r16_c1_r;
    __m256 left_r16_c0_i, left_r16_c1_i;
    __m256 left_r17_c0_r, left_r17_c1_r;
    __m256 left_r17_c0_i, left_r17_c1_i;
    __m256 left_r18_c0_r, left_r18_c1_r;
    __m256 left_r18_c0_i, left_r18_c1_i;
    __m256 right_r_i, right_i_r;
    SAL_i32 c_c;
    SAL_i32 strip_start_col;
    SAL_i32 next_strip_col;
#ifdef IACA_LOAD_LEFT
    IACA_START
#endif

    left_r0_c0_r = _mm256_broadcast_ss (&(A[0] + 0)->real);
    left_r0_c0_i = _mm256_broadcast_ss (&(A[0] + 0)->imag);
    left_r0_c1_r = _mm256_broadcast_ss (&(A[0] + 1)->real);
    left_r0_c1_i = _mm256_broadcast_ss (&(A[0] + 1)->imag);

    left_r1_c0_r = _mm256_broadcast_ss (&(A[1] + 0)->real);
    left_r1_c0_i = _mm256_broadcast_ss (&(A[1] + 0)->imag);
    left_r1_c1_r = _mm256_broadcast_ss (&(A[1] + 1)->real);
    left_r1_c1_i = _mm256_broadcast_ss (&(A[1] + 1)->imag);

    left_r2_c0_r = _mm256_broadcast_ss (&(A[2] + 0)->real);
    left_r2_c0_i = _mm256_broadcast_ss (&(A[2] + 0)->imag);
    left_r2_c1_r = _mm256_broadcast_ss (&(A[2] + 1)->real);
    left_r2_c1_i = _mm256_broadcast_ss (&(A[2] + 1)->imag);

    left_r3_c0_r = _mm256_broadcast_ss (&(A[3] + 0)->real);
    left_r3_c0_i = _mm256_broadcast_ss (&(A[3] + 0)->imag);
    left_r3_c1_r = _mm256_broadcast_ss (&(A[3] + 1)->real);
    left_r3_c1_i = _mm256_broadcast_ss (&(A[3] + 1)->imag);

    left_r4_c0_r = _mm256_broadcast_ss (&(A[4] + 0)->real);
    left_r4_c0_i = _mm256_broadcast_ss (&(A[4] + 0)->imag);
    left_r4_c1_r = _mm256_broadcast_ss (&(A[4] + 1)->real);
    left_r4_c1_i = _mm256_broadcast_ss (&(A[4] + 1)->imag);

    left_r5_c0_r = _mm256_broadcast_ss (&(A[5] + 0)->real);
    left_r5_c0_i = _mm256_broadcast_ss (&(A[5] + 0)->imag);
    left_r5_c1_r = _mm256_broadcast_ss (&(A[5] + 1)->real);
    left_r5_c1_i = _mm256_broadcast_ss (&(A[5] + 1)->imag);

    left_r6_c0_r = _mm256_broadcast_ss (&(A[6] + 0)->real);
    left_r6_c0_i = _mm256_broadcast_ss (&(A[6] + 0)->imag);
    left_r6_c1_r = _mm256_broadcast_ss (&(A[6] + 1)->real);
    left_r6_c1_i = _mm256_broadcast_ss (&(A[6] + 1)->imag);

    left_r7_c0_r = _mm256_broadcast_ss (&(A[7] + 0)->real);
    left_r7_c0_i = _mm256_broadcast_ss (&(A[7] + 0)->imag);
    left_r7_c1_r = _mm256_broadcast_ss (&(A[7] + 1)->real);
    left_r7_c1_i = _mm256_broadcast_ss (&(A[7] + 1)->imag);

    left_r8_c0_r = _mm256_broadcast_ss (&(A[8] + 0)->real);
    left_r8_c0_i = _mm256_broadcast_ss (&(A[8] + 0)->imag);
    left_r8_c1_r = _mm256_broadcast_ss (&(A[8] + 1)->real);
    left_r8_c1_i = _mm256_broadcast_ss (&(A[8] + 1)->imag);

    left_r9_c0_r = _mm256_broadcast_ss (&(A[9] + 0)->real);
    left_r9_c0_i = _mm256_broadcast_ss (&(A[9] + 0)->imag);
    left_r9_c1_r = _mm256_broadcast_ss (&(A[9] + 1)->real);
    left_r9_c1_i = _mm256_broadcast_ss (&(A[9] + 1)->imag);

    left_r10_c0_r = _mm256_broadcast_ss (&(A[10] + 0)->real);
    left_r10_c0_i = _mm256_broadcast_ss (&(A[10] + 0)->imag);
    left_r10_c1_r = _mm256_broadcast_ss (&(A[10] + 1)->real);
    left_r10_c1_i = _mm256_broadcast_ss (&(A[10] + 1)->imag);

    left_r11_c0_r = _mm256_broadcast_ss (&(A[11] + 0)->real);
    left_r11_c0_i = _mm256_broadcast_ss (&(A[11] + 0)->imag);
    left_r11_c1_r = _mm256_broadcast_ss (&(A[11] + 1)->real);
    left_r11_c1_i = _mm256_broadcast_ss (&(A[11] + 1)->imag);

    left_r12_c0_r = _mm256_broadcast_ss (&(A[12] + 0)->real);
    left_r12_c0_i = _mm256_broadcast_ss (&(A[12] + 0)->imag);
    left_r12_c1_r = _mm256_broadcast_ss (&(A[12] + 1)->real);
    left_r12_c1_i = _mm256_broadcast_ss (&(A[12] + 1)->imag);

    left_r13_c0_r = _mm256_broadcast_ss (&(A[13] + 0)->real);
    left_r13_c0_i = _mm256_broadcast_ss (&(A[13] + 0)->imag);
    left_r13_c1_r = _mm256_broadcast_ss (&(A[13] + 1)->real);
    left_r13_c1_i = _mm256_broadcast_ss (&(A[13] + 1)->imag);

    left_r14_c0_r = _mm256_broadcast_ss (&(A[14] + 0)->real);
    left_r14_c0_i = _mm256_broadcast_ss (&(A[14] + 0)->imag);
    left_r14_c1_r = _mm256_broadcast_ss (&(A[14] + 1)->real);
    left_r14_c1_i = _mm256_broadcast_ss (&(A[14] + 1)->imag);

    left_r15_c0_r = _mm256_broadcast_ss (&(A[15] + 0)->real);
    left_r15_c0_i = _mm256_broadcast_ss (&(A[15] + 0)->imag);
    left_r15_c1_r = _mm256_broadcast_ss (&(A[15] + 1)->real);
    left_r15_c1_i = _mm256_broadcast_ss (&(A[15] + 1)->imag);

    left_r16_c0_r = _mm256_broadcast_ss (&(A[16] + 0)->real);
    left_r16_c0_i = _mm256_broadcast_ss (&(A[16] + 0)->imag);
    left_r16_c1_r = _mm256_broadcast_ss (&(A[16] + 1)->real);
    left_r16_c1_i = _mm256_broadcast_ss (&(A[16] + 1)->imag);

    left_r17_c0_r = _mm256_broadcast_ss (&(A[17] + 0)->real);
    left_r17_c0_i = _mm256_broadcast_ss (&(A[17] + 0)->imag);
    left_r17_c1_r = _mm256_broadcast_ss (&(A[17] + 1)->real);
    left_r17_c1_i = _mm256_broadcast_ss (&(A[17] + 1)->imag);

    left_r18_c0_r = _mm256_broadcast_ss (&(A[18] + 0)->real);
    left_r18_c0_i = _mm256_broadcast_ss (&(A[18] + 0)->imag);
    left_r18_c1_r = _mm256_broadcast_ss (&(A[18] + 1)->real);
    left_r18_c1_i = _mm256_broadcast_ss (&(A[18] + 1)->imag);
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    strip_start_col = 0;
    while (strip_start_col < nc_c)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
        next_strip_col = strip_start_col + strip_size_samples;
        if (next_strip_col > nc_c)
        {
            next_strip_col = nc_c;
        }

        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[0] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r0_c0_i)));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r1_c0_i)));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r2_c0_i)));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r3_c0_i)));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r4_c0_i)));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r5_c0_i)));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r6_c0_i)));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r7_c0_i)));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r8_c0_i)));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c0_r),
                                                                    _mm256_mul_ps (right_i_r, left_r9_c0_i)));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r10_c0_i)));
            _mm256_store_ps (&(C[11] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r11_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r11_c0_i)));
            _mm256_store_ps (&(C[12] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r12_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r12_c0_i)));
            _mm256_store_ps (&(C[13] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r13_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r13_c0_i)));
            _mm256_store_ps (&(C[14] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r14_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r14_c0_i)));
            _mm256_store_ps (&(C[15] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r15_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r15_c0_i)));
            _mm256_store_ps (&(C[16] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r16_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r16_c0_i)));
            _mm256_store_ps (&(C[17] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r17_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r17_c0_i)));
            _mm256_store_ps (&(C[18] + c_c)->real, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r18_c0_r),
                                                                     _mm256_mul_ps (right_i_r, left_r18_c0_i)));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[1] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c1_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c1_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c1_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c1_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c1_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c1_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c1_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c1_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c1_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c1_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c1_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c1_i))));
            _mm256_store_ps (&(C[11] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[11] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r11_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r11_c1_i))));
            _mm256_store_ps (&(C[12] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[12] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r12_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r12_c1_i))));
            _mm256_store_ps (&(C[13] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[13] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r13_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r13_c1_i))));
            _mm256_store_ps (&(C[14] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[14] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r14_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r14_c1_i))));
            _mm256_store_ps (&(C[15] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[15] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r15_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r15_c1_i))));
            _mm256_store_ps (&(C[16] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[16] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r16_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r16_c1_i))));
            _mm256_store_ps (&(C[17] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[17] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r17_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r17_c1_i))));
            _mm256_store_ps (&(C[18] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[18] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r18_c1_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r18_c1_i))));
        }
        strip_start_col = next_strip_col;
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

