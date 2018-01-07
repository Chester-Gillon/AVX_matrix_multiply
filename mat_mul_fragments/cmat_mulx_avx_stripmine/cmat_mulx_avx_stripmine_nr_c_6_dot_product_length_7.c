/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_stripmine_nr_c_6_dot_product_length_7 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                          SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                          SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                          SAL_i32 nc_c)    /* column count in C */
{
    const SAL_i32 nr_c = 6;
    /*const SAL_i32 dot_product_length = 7;*/
    const SAL_i32 L1_cache_size = 32768;
    const SAL_i32 strip_size_samples = (L1_cache_size / sizeof(__m256) / (nr_c + 2)) / 4 * 4;
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r, left_r3_c5_r, left_r3_c6_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i, left_r3_c5_i, left_r3_c6_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r, left_r4_c4_r, left_r4_c5_r, left_r4_c6_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i, left_r4_c4_i, left_r4_c5_i, left_r4_c6_i;
    __m256 left_r5_c0_r, left_r5_c1_r, left_r5_c2_r, left_r5_c3_r, left_r5_c4_r, left_r5_c5_r, left_r5_c6_r;
    __m256 left_r5_c0_i, left_r5_c1_i, left_r5_c2_i, left_r5_c3_i, left_r5_c4_i, left_r5_c5_i, left_r5_c6_i;
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

    left_r4_c0_r = _mm256_broadcast_ss (&(A[4] + 0)->real);
    left_r4_c0_i = _mm256_broadcast_ss (&(A[4] + 0)->imag);
    left_r4_c1_r = _mm256_broadcast_ss (&(A[4] + 1)->real);
    left_r4_c1_i = _mm256_broadcast_ss (&(A[4] + 1)->imag);
    left_r4_c2_r = _mm256_broadcast_ss (&(A[4] + 2)->real);
    left_r4_c2_i = _mm256_broadcast_ss (&(A[4] + 2)->imag);
    left_r4_c3_r = _mm256_broadcast_ss (&(A[4] + 3)->real);
    left_r4_c3_i = _mm256_broadcast_ss (&(A[4] + 3)->imag);
    left_r4_c4_r = _mm256_broadcast_ss (&(A[4] + 4)->real);
    left_r4_c4_i = _mm256_broadcast_ss (&(A[4] + 4)->imag);
    left_r4_c5_r = _mm256_broadcast_ss (&(A[4] + 5)->real);
    left_r4_c5_i = _mm256_broadcast_ss (&(A[4] + 5)->imag);
    left_r4_c6_r = _mm256_broadcast_ss (&(A[4] + 6)->real);
    left_r4_c6_i = _mm256_broadcast_ss (&(A[4] + 6)->imag);

    left_r5_c0_r = _mm256_broadcast_ss (&(A[5] + 0)->real);
    left_r5_c0_i = _mm256_broadcast_ss (&(A[5] + 0)->imag);
    left_r5_c1_r = _mm256_broadcast_ss (&(A[5] + 1)->real);
    left_r5_c1_i = _mm256_broadcast_ss (&(A[5] + 1)->imag);
    left_r5_c2_r = _mm256_broadcast_ss (&(A[5] + 2)->real);
    left_r5_c2_i = _mm256_broadcast_ss (&(A[5] + 2)->imag);
    left_r5_c3_r = _mm256_broadcast_ss (&(A[5] + 3)->real);
    left_r5_c3_i = _mm256_broadcast_ss (&(A[5] + 3)->imag);
    left_r5_c4_r = _mm256_broadcast_ss (&(A[5] + 4)->real);
    left_r5_c4_i = _mm256_broadcast_ss (&(A[5] + 4)->imag);
    left_r5_c5_r = _mm256_broadcast_ss (&(A[5] + 5)->real);
    left_r5_c5_i = _mm256_broadcast_ss (&(A[5] + 5)->imag);
    left_r5_c6_r = _mm256_broadcast_ss (&(A[5] + 6)->real);
    left_r5_c6_i = _mm256_broadcast_ss (&(A[5] + 6)->imag);
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
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[2] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c2_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c2_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c2_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c2_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c2_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c2_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[3] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c3_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c3_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c3_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c3_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c3_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c3_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[4] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c4_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c4_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c4_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c4_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c4_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c4_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[5] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c5_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c5_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c5_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c5_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c5_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c5_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[6] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c6_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c6_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c6_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c6_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c6_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c6_i))));
        }
        strip_start_col = next_strip_col;
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

