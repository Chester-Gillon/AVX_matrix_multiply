/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_nr_c_3_dot_product_length_3 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i;
    __m256 right_r0_r_i, right_r0_i_r;
    __m256 right_r1_r_i, right_r1_i_r;
    __m256 right_r2_r_i, right_r2_i_r;
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

    left_r1_c0_r = _mm256_broadcast_ss (&(A[1] + 0)->real);
    left_r1_c0_i = _mm256_broadcast_ss (&(A[1] + 0)->imag);
    left_r1_c1_r = _mm256_broadcast_ss (&(A[1] + 1)->real);
    left_r1_c1_i = _mm256_broadcast_ss (&(A[1] + 1)->imag);
    left_r1_c2_r = _mm256_broadcast_ss (&(A[1] + 2)->real);
    left_r1_c2_i = _mm256_broadcast_ss (&(A[1] + 2)->imag);

    left_r2_c0_r = _mm256_broadcast_ss (&(A[2] + 0)->real);
    left_r2_c0_i = _mm256_broadcast_ss (&(A[2] + 0)->imag);
    left_r2_c1_r = _mm256_broadcast_ss (&(A[2] + 1)->real);
    left_r2_c1_i = _mm256_broadcast_ss (&(A[2] + 1)->imag);
    left_r2_c2_r = _mm256_broadcast_ss (&(A[2] + 2)->real);
    left_r2_c2_i = _mm256_broadcast_ss (&(A[2] + 2)->imag);
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
        right_r0_r_i = _mm256_load_ps (&(B[0] + c_c)->real);
        right_r0_i_r = _mm256_permute_ps (right_r0_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r1_r_i = _mm256_load_ps (&(B[1] + c_c)->real);
        right_r1_i_r = _mm256_permute_ps (right_r1_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r2_r_i = _mm256_load_ps (&(B[2] + c_c)->real);
        right_r2_i_r = _mm256_permute_ps (right_r2_r_i, SWAP_REAL_IMAG_PERMUTE);

        _mm256_store_ps (&(C[0] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r0_c0_r),
                                      _mm256_mul_ps (right_r0_i_r, left_r0_c0_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r0_c1_r),
                                      _mm256_mul_ps (right_r1_i_r, left_r0_c1_i))),
                _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r0_c2_r),
                                  _mm256_mul_ps (right_r2_i_r, left_r0_c2_i))));

        _mm256_store_ps (&(C[1] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r1_c0_r),
                                      _mm256_mul_ps (right_r0_i_r, left_r1_c0_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r1_c1_r),
                                      _mm256_mul_ps (right_r1_i_r, left_r1_c1_i))),
                _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r1_c2_r),
                                  _mm256_mul_ps (right_r2_i_r, left_r1_c2_i))));

        _mm256_store_ps (&(C[2] + c_c)->real,
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r2_c0_r),
                                      _mm256_mul_ps (right_r0_i_r, left_r2_c0_i)),
                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r2_c1_r),
                                      _mm256_mul_ps (right_r1_i_r, left_r2_c1_i))),
                _mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r2_c2_r),
                                  _mm256_mul_ps (right_r2_i_r, left_r2_c2_i))));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

