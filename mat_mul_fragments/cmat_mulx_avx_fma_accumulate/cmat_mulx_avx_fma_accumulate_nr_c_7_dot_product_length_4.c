/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_fma_accumulate_nr_c_7_dot_product_length_4 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                               SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                               SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                               SAL_i32 nc_c)    /* column count in C */
{
    const __m256 negate_even_mult = _mm256_set_ps (1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i;
    __m256 left_r5_c0_r, left_r5_c1_r, left_r5_c2_r, left_r5_c3_r;
    __m256 left_r5_c0_i, left_r5_c1_i, left_r5_c2_i, left_r5_c3_i;
    __m256 left_r6_c0_r, left_r6_c1_r, left_r6_c2_r, left_r6_c3_r;
    __m256 left_r6_c0_i, left_r6_c1_i, left_r6_c2_i, left_r6_c3_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    __m256 output_r3;
    __m256 output_r4;
    __m256 output_r5;
    __m256 output_r6;
    SAL_i32 c_c;
#ifdef IACA_LOAD_LEFT
    IACA_START
#endif

    left_r0_c0_r = _mm256_broadcast_ss (&(A[0] + 0)->real);
    left_r0_c0_i = _mm256_broadcast_ss (&(A[0] + 0)->imag);
    left_r0_c1_r = _mm256_broadcast_ss (&(A[0] + 1)->real);
    left_r0_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 1)->imag));
    left_r0_c2_r = _mm256_broadcast_ss (&(A[0] + 2)->real);
    left_r0_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 2)->imag));
    left_r0_c3_r = _mm256_broadcast_ss (&(A[0] + 3)->real);
    left_r0_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 3)->imag));

    left_r1_c0_r = _mm256_broadcast_ss (&(A[1] + 0)->real);
    left_r1_c0_i = _mm256_broadcast_ss (&(A[1] + 0)->imag);
    left_r1_c1_r = _mm256_broadcast_ss (&(A[1] + 1)->real);
    left_r1_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 1)->imag));
    left_r1_c2_r = _mm256_broadcast_ss (&(A[1] + 2)->real);
    left_r1_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 2)->imag));
    left_r1_c3_r = _mm256_broadcast_ss (&(A[1] + 3)->real);
    left_r1_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 3)->imag));

    left_r2_c0_r = _mm256_broadcast_ss (&(A[2] + 0)->real);
    left_r2_c0_i = _mm256_broadcast_ss (&(A[2] + 0)->imag);
    left_r2_c1_r = _mm256_broadcast_ss (&(A[2] + 1)->real);
    left_r2_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 1)->imag));
    left_r2_c2_r = _mm256_broadcast_ss (&(A[2] + 2)->real);
    left_r2_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 2)->imag));
    left_r2_c3_r = _mm256_broadcast_ss (&(A[2] + 3)->real);
    left_r2_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 3)->imag));

    left_r3_c0_r = _mm256_broadcast_ss (&(A[3] + 0)->real);
    left_r3_c0_i = _mm256_broadcast_ss (&(A[3] + 0)->imag);
    left_r3_c1_r = _mm256_broadcast_ss (&(A[3] + 1)->real);
    left_r3_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 1)->imag));
    left_r3_c2_r = _mm256_broadcast_ss (&(A[3] + 2)->real);
    left_r3_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 2)->imag));
    left_r3_c3_r = _mm256_broadcast_ss (&(A[3] + 3)->real);
    left_r3_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 3)->imag));

    left_r4_c0_r = _mm256_broadcast_ss (&(A[4] + 0)->real);
    left_r4_c0_i = _mm256_broadcast_ss (&(A[4] + 0)->imag);
    left_r4_c1_r = _mm256_broadcast_ss (&(A[4] + 1)->real);
    left_r4_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 1)->imag));
    left_r4_c2_r = _mm256_broadcast_ss (&(A[4] + 2)->real);
    left_r4_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 2)->imag));
    left_r4_c3_r = _mm256_broadcast_ss (&(A[4] + 3)->real);
    left_r4_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 3)->imag));

    left_r5_c0_r = _mm256_broadcast_ss (&(A[5] + 0)->real);
    left_r5_c0_i = _mm256_broadcast_ss (&(A[5] + 0)->imag);
    left_r5_c1_r = _mm256_broadcast_ss (&(A[5] + 1)->real);
    left_r5_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 1)->imag));
    left_r5_c2_r = _mm256_broadcast_ss (&(A[5] + 2)->real);
    left_r5_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 2)->imag));
    left_r5_c3_r = _mm256_broadcast_ss (&(A[5] + 3)->real);
    left_r5_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 3)->imag));

    left_r6_c0_r = _mm256_broadcast_ss (&(A[6] + 0)->real);
    left_r6_c0_i = _mm256_broadcast_ss (&(A[6] + 0)->imag);
    left_r6_c1_r = _mm256_broadcast_ss (&(A[6] + 1)->real);
    left_r6_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 1)->imag));
    left_r6_c2_r = _mm256_broadcast_ss (&(A[6] + 2)->real);
    left_r6_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 2)->imag));
    left_r6_c3_r = _mm256_broadcast_ss (&(A[6] + 3)->real);
    left_r6_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 3)->imag));
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
        output_r0 = _mm256_fmaddsub_ps (               right_r_i, left_r0_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_fmaddsub_ps (               right_r_i, left_r1_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_fmaddsub_ps (               right_r_i, left_r2_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r2_c0_i));
        output_r3 = _mm256_fmaddsub_ps (               right_r_i, left_r3_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r3_c0_i));
        output_r4 = _mm256_fmaddsub_ps (               right_r_i, left_r4_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r4_c0_i));
        output_r5 = _mm256_fmaddsub_ps (               right_r_i, left_r5_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r5_c0_i));
        output_r6 = _mm256_fmaddsub_ps (               right_r_i, left_r6_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r6_c0_i));
        right_r_i = _mm256_load_ps (&(B[1] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c1_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c1_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c1_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c1_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c1_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c1_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c1_i, output_r6));
        right_r_i = _mm256_load_ps (&(B[2] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c2_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c2_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c2_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c2_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c2_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c2_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c2_i, output_r6));
        right_r_i = _mm256_load_ps (&(B[3] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&(C[0] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r0_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r0_c3_i, output_r0)));
        _mm256_store_ps (&(C[1] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r1_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r1_c3_i, output_r1)));
        _mm256_store_ps (&(C[2] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r2_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r2_c3_i, output_r2)));
        _mm256_store_ps (&(C[3] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r3_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r3_c3_i, output_r3)));
        _mm256_store_ps (&(C[4] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r4_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r4_c3_i, output_r4)));
        _mm256_store_ps (&(C[5] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r5_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r5_c3_i, output_r5)));
        _mm256_store_ps (&(C[6] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r6_c3_r,
                                                               _mm256_fmadd_ps (right_i_r, left_r6_c3_i, output_r6)));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

