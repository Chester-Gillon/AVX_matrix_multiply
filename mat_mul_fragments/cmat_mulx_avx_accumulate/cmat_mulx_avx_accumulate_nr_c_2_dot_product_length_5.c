/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_accumulate_nr_c_2_dot_product_length_5 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                           SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                           SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
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
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
        right_r_i = _mm256_loadu_ps (&(B[0] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                      _mm256_mul_ps (right_i_r, left_r1_c0_i));
        right_r_i = _mm256_loadu_ps (&(B[1] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        right_r_i = _mm256_loadu_ps (&(B[2] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        right_r_i = _mm256_loadu_ps (&(B[3] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        right_r_i = _mm256_loadu_ps (&(B[4] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_storeu_ps (&(C[0] + c_c)->real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                                           _mm256_mul_ps (right_i_r, left_r0_c4_i))));
        _mm256_storeu_ps (&(C[1] + c_c)->real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                                           _mm256_mul_ps (right_i_r, left_r1_c4_i))));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

