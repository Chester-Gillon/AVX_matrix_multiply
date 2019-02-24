/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_nr_c_12_dot_product_length_5 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                 SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                 SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                 SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r, left_r4_c4_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i, left_r4_c4_i;
    __m256 left_r5_c0_r, left_r5_c1_r, left_r5_c2_r, left_r5_c3_r, left_r5_c4_r;
    __m256 left_r5_c0_i, left_r5_c1_i, left_r5_c2_i, left_r5_c3_i, left_r5_c4_i;
    __m256 left_r6_c0_r, left_r6_c1_r, left_r6_c2_r, left_r6_c3_r, left_r6_c4_r;
    __m256 left_r6_c0_i, left_r6_c1_i, left_r6_c2_i, left_r6_c3_i, left_r6_c4_i;
    __m256 left_r7_c0_r, left_r7_c1_r, left_r7_c2_r, left_r7_c3_r, left_r7_c4_r;
    __m256 left_r7_c0_i, left_r7_c1_i, left_r7_c2_i, left_r7_c3_i, left_r7_c4_i;
    __m256 left_r8_c0_r, left_r8_c1_r, left_r8_c2_r, left_r8_c3_r, left_r8_c4_r;
    __m256 left_r8_c0_i, left_r8_c1_i, left_r8_c2_i, left_r8_c3_i, left_r8_c4_i;
    __m256 left_r9_c0_r, left_r9_c1_r, left_r9_c2_r, left_r9_c3_r, left_r9_c4_r;
    __m256 left_r9_c0_i, left_r9_c1_i, left_r9_c2_i, left_r9_c3_i, left_r9_c4_i;
    __m256 left_r10_c0_r, left_r10_c1_r, left_r10_c2_r, left_r10_c3_r, left_r10_c4_r;
    __m256 left_r10_c0_i, left_r10_c1_i, left_r10_c2_i, left_r10_c3_i, left_r10_c4_i;
    __m256 left_r11_c0_r, left_r11_c1_r, left_r11_c2_r, left_r11_c3_r, left_r11_c4_r;
    __m256 left_r11_c0_i, left_r11_c1_i, left_r11_c2_i, left_r11_c3_i, left_r11_c4_i;
    __m256 right_r0_r_i, right_r0_i_r;
    __m256 right_r1_r_i, right_r1_i_r;
    __m256 right_r2_r_i, right_r2_i_r;
    __m256 right_r3_r_i, right_r3_i_r;
    __m256 right_r4_r_i, right_r4_i_r;
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

    left_r6_c0_r = _mm256_broadcast_ss (&(A[6] + 0)->real);
    left_r6_c0_i = _mm256_broadcast_ss (&(A[6] + 0)->imag);
    left_r6_c1_r = _mm256_broadcast_ss (&(A[6] + 1)->real);
    left_r6_c1_i = _mm256_broadcast_ss (&(A[6] + 1)->imag);
    left_r6_c2_r = _mm256_broadcast_ss (&(A[6] + 2)->real);
    left_r6_c2_i = _mm256_broadcast_ss (&(A[6] + 2)->imag);
    left_r6_c3_r = _mm256_broadcast_ss (&(A[6] + 3)->real);
    left_r6_c3_i = _mm256_broadcast_ss (&(A[6] + 3)->imag);
    left_r6_c4_r = _mm256_broadcast_ss (&(A[6] + 4)->real);
    left_r6_c4_i = _mm256_broadcast_ss (&(A[6] + 4)->imag);

    left_r7_c0_r = _mm256_broadcast_ss (&(A[7] + 0)->real);
    left_r7_c0_i = _mm256_broadcast_ss (&(A[7] + 0)->imag);
    left_r7_c1_r = _mm256_broadcast_ss (&(A[7] + 1)->real);
    left_r7_c1_i = _mm256_broadcast_ss (&(A[7] + 1)->imag);
    left_r7_c2_r = _mm256_broadcast_ss (&(A[7] + 2)->real);
    left_r7_c2_i = _mm256_broadcast_ss (&(A[7] + 2)->imag);
    left_r7_c3_r = _mm256_broadcast_ss (&(A[7] + 3)->real);
    left_r7_c3_i = _mm256_broadcast_ss (&(A[7] + 3)->imag);
    left_r7_c4_r = _mm256_broadcast_ss (&(A[7] + 4)->real);
    left_r7_c4_i = _mm256_broadcast_ss (&(A[7] + 4)->imag);

    left_r8_c0_r = _mm256_broadcast_ss (&(A[8] + 0)->real);
    left_r8_c0_i = _mm256_broadcast_ss (&(A[8] + 0)->imag);
    left_r8_c1_r = _mm256_broadcast_ss (&(A[8] + 1)->real);
    left_r8_c1_i = _mm256_broadcast_ss (&(A[8] + 1)->imag);
    left_r8_c2_r = _mm256_broadcast_ss (&(A[8] + 2)->real);
    left_r8_c2_i = _mm256_broadcast_ss (&(A[8] + 2)->imag);
    left_r8_c3_r = _mm256_broadcast_ss (&(A[8] + 3)->real);
    left_r8_c3_i = _mm256_broadcast_ss (&(A[8] + 3)->imag);
    left_r8_c4_r = _mm256_broadcast_ss (&(A[8] + 4)->real);
    left_r8_c4_i = _mm256_broadcast_ss (&(A[8] + 4)->imag);

    left_r9_c0_r = _mm256_broadcast_ss (&(A[9] + 0)->real);
    left_r9_c0_i = _mm256_broadcast_ss (&(A[9] + 0)->imag);
    left_r9_c1_r = _mm256_broadcast_ss (&(A[9] + 1)->real);
    left_r9_c1_i = _mm256_broadcast_ss (&(A[9] + 1)->imag);
    left_r9_c2_r = _mm256_broadcast_ss (&(A[9] + 2)->real);
    left_r9_c2_i = _mm256_broadcast_ss (&(A[9] + 2)->imag);
    left_r9_c3_r = _mm256_broadcast_ss (&(A[9] + 3)->real);
    left_r9_c3_i = _mm256_broadcast_ss (&(A[9] + 3)->imag);
    left_r9_c4_r = _mm256_broadcast_ss (&(A[9] + 4)->real);
    left_r9_c4_i = _mm256_broadcast_ss (&(A[9] + 4)->imag);

    left_r10_c0_r = _mm256_broadcast_ss (&(A[10] + 0)->real);
    left_r10_c0_i = _mm256_broadcast_ss (&(A[10] + 0)->imag);
    left_r10_c1_r = _mm256_broadcast_ss (&(A[10] + 1)->real);
    left_r10_c1_i = _mm256_broadcast_ss (&(A[10] + 1)->imag);
    left_r10_c2_r = _mm256_broadcast_ss (&(A[10] + 2)->real);
    left_r10_c2_i = _mm256_broadcast_ss (&(A[10] + 2)->imag);
    left_r10_c3_r = _mm256_broadcast_ss (&(A[10] + 3)->real);
    left_r10_c3_i = _mm256_broadcast_ss (&(A[10] + 3)->imag);
    left_r10_c4_r = _mm256_broadcast_ss (&(A[10] + 4)->real);
    left_r10_c4_i = _mm256_broadcast_ss (&(A[10] + 4)->imag);

    left_r11_c0_r = _mm256_broadcast_ss (&(A[11] + 0)->real);
    left_r11_c0_i = _mm256_broadcast_ss (&(A[11] + 0)->imag);
    left_r11_c1_r = _mm256_broadcast_ss (&(A[11] + 1)->real);
    left_r11_c1_i = _mm256_broadcast_ss (&(A[11] + 1)->imag);
    left_r11_c2_r = _mm256_broadcast_ss (&(A[11] + 2)->real);
    left_r11_c2_i = _mm256_broadcast_ss (&(A[11] + 2)->imag);
    left_r11_c3_r = _mm256_broadcast_ss (&(A[11] + 3)->real);
    left_r11_c3_i = _mm256_broadcast_ss (&(A[11] + 3)->imag);
    left_r11_c4_r = _mm256_broadcast_ss (&(A[11] + 4)->real);
    left_r11_c4_i = _mm256_broadcast_ss (&(A[11] + 4)->imag);
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
        right_r0_r_i = _mm256_loadu_ps (&(B[0] + c_c)->real);
        right_r0_i_r = _mm256_permute_ps (right_r0_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r1_r_i = _mm256_loadu_ps (&(B[1] + c_c)->real);
        right_r1_i_r = _mm256_permute_ps (right_r1_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r2_r_i = _mm256_loadu_ps (&(B[2] + c_c)->real);
        right_r2_i_r = _mm256_permute_ps (right_r2_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r3_r_i = _mm256_loadu_ps (&(B[3] + c_c)->real);
        right_r3_i_r = _mm256_permute_ps (right_r3_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r4_r_i = _mm256_loadu_ps (&(B[4] + c_c)->real);
        right_r4_i_r = _mm256_permute_ps (right_r4_r_i, SWAP_REAL_IMAG_PERMUTE);

        _mm256_storeu_ps (&(C[0] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r0_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r0_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r0_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r0_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r0_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r0_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r0_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r0_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r0_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r0_c4_i)))));

        _mm256_storeu_ps (&(C[1] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r1_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r1_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r1_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r1_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r1_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r1_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r1_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r1_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r1_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r1_c4_i)))));

        _mm256_storeu_ps (&(C[2] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r2_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r2_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r2_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r2_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r2_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r2_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r2_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r2_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r2_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r2_c4_i)))));

        _mm256_storeu_ps (&(C[3] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r3_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r3_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r3_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r3_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r3_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r3_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r3_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r3_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r3_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r3_c4_i)))));

        _mm256_storeu_ps (&(C[4] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r4_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r4_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r4_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r4_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r4_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r4_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r4_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r4_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r4_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r4_c4_i)))));

        _mm256_storeu_ps (&(C[5] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r5_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r5_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r5_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r5_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r5_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r5_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r5_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r5_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r5_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r5_c4_i)))));

        _mm256_storeu_ps (&(C[6] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r6_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r6_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r6_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r6_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r6_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r6_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r6_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r6_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r6_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r6_c4_i)))));

        _mm256_storeu_ps (&(C[7] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r7_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r7_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r7_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r7_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r7_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r7_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r7_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r7_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r7_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r7_c4_i)))));

        _mm256_storeu_ps (&(C[8] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r8_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r8_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r8_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r8_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r8_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r8_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r8_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r8_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r8_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r8_c4_i)))));

        _mm256_storeu_ps (&(C[9] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r9_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r9_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r9_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r9_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r9_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r9_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r9_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r9_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r9_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r9_c4_i)))));

        _mm256_storeu_ps (&(C[10] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r10_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r10_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r10_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r10_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r10_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r10_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r10_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r10_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r10_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r10_c4_i)))));

        _mm256_storeu_ps (&(C[11] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r11_c0_r),
                                          _mm256_mul_ps (right_r0_i_r, left_r11_c0_i)),
                        _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r11_c1_r),
                                          _mm256_mul_ps (right_r1_i_r, left_r11_c1_i))),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r11_c2_r),
                                      _mm256_mul_ps (right_r2_i_r, left_r11_c2_i))),
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r11_c3_r),
                                      _mm256_mul_ps (right_r3_i_r, left_r11_c3_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r11_c4_r),
                                      _mm256_mul_ps (right_r4_i_r, left_r11_c4_i)))));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

