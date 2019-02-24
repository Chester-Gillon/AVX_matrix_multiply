/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_fma_accumulate_nr_c_17_dot_product_length_19 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                 SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                 SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                 SAL_i32 nc_c)    /* column count in C */
{
    const __m256 negate_even_mult = _mm256_set_ps (1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r, left_r0_c17_r, left_r0_c18_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i, left_r0_c17_i, left_r0_c18_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r, left_r1_c17_r, left_r1_c18_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i, left_r1_c17_i, left_r1_c18_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r, left_r2_c17_r, left_r2_c18_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i, left_r2_c17_i, left_r2_c18_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r, left_r3_c5_r, left_r3_c6_r, left_r3_c7_r, left_r3_c8_r, left_r3_c9_r, left_r3_c10_r, left_r3_c11_r, left_r3_c12_r, left_r3_c13_r, left_r3_c14_r, left_r3_c15_r, left_r3_c16_r, left_r3_c17_r, left_r3_c18_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i, left_r3_c5_i, left_r3_c6_i, left_r3_c7_i, left_r3_c8_i, left_r3_c9_i, left_r3_c10_i, left_r3_c11_i, left_r3_c12_i, left_r3_c13_i, left_r3_c14_i, left_r3_c15_i, left_r3_c16_i, left_r3_c17_i, left_r3_c18_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r, left_r4_c4_r, left_r4_c5_r, left_r4_c6_r, left_r4_c7_r, left_r4_c8_r, left_r4_c9_r, left_r4_c10_r, left_r4_c11_r, left_r4_c12_r, left_r4_c13_r, left_r4_c14_r, left_r4_c15_r, left_r4_c16_r, left_r4_c17_r, left_r4_c18_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i, left_r4_c4_i, left_r4_c5_i, left_r4_c6_i, left_r4_c7_i, left_r4_c8_i, left_r4_c9_i, left_r4_c10_i, left_r4_c11_i, left_r4_c12_i, left_r4_c13_i, left_r4_c14_i, left_r4_c15_i, left_r4_c16_i, left_r4_c17_i, left_r4_c18_i;
    __m256 left_r5_c0_r, left_r5_c1_r, left_r5_c2_r, left_r5_c3_r, left_r5_c4_r, left_r5_c5_r, left_r5_c6_r, left_r5_c7_r, left_r5_c8_r, left_r5_c9_r, left_r5_c10_r, left_r5_c11_r, left_r5_c12_r, left_r5_c13_r, left_r5_c14_r, left_r5_c15_r, left_r5_c16_r, left_r5_c17_r, left_r5_c18_r;
    __m256 left_r5_c0_i, left_r5_c1_i, left_r5_c2_i, left_r5_c3_i, left_r5_c4_i, left_r5_c5_i, left_r5_c6_i, left_r5_c7_i, left_r5_c8_i, left_r5_c9_i, left_r5_c10_i, left_r5_c11_i, left_r5_c12_i, left_r5_c13_i, left_r5_c14_i, left_r5_c15_i, left_r5_c16_i, left_r5_c17_i, left_r5_c18_i;
    __m256 left_r6_c0_r, left_r6_c1_r, left_r6_c2_r, left_r6_c3_r, left_r6_c4_r, left_r6_c5_r, left_r6_c6_r, left_r6_c7_r, left_r6_c8_r, left_r6_c9_r, left_r6_c10_r, left_r6_c11_r, left_r6_c12_r, left_r6_c13_r, left_r6_c14_r, left_r6_c15_r, left_r6_c16_r, left_r6_c17_r, left_r6_c18_r;
    __m256 left_r6_c0_i, left_r6_c1_i, left_r6_c2_i, left_r6_c3_i, left_r6_c4_i, left_r6_c5_i, left_r6_c6_i, left_r6_c7_i, left_r6_c8_i, left_r6_c9_i, left_r6_c10_i, left_r6_c11_i, left_r6_c12_i, left_r6_c13_i, left_r6_c14_i, left_r6_c15_i, left_r6_c16_i, left_r6_c17_i, left_r6_c18_i;
    __m256 left_r7_c0_r, left_r7_c1_r, left_r7_c2_r, left_r7_c3_r, left_r7_c4_r, left_r7_c5_r, left_r7_c6_r, left_r7_c7_r, left_r7_c8_r, left_r7_c9_r, left_r7_c10_r, left_r7_c11_r, left_r7_c12_r, left_r7_c13_r, left_r7_c14_r, left_r7_c15_r, left_r7_c16_r, left_r7_c17_r, left_r7_c18_r;
    __m256 left_r7_c0_i, left_r7_c1_i, left_r7_c2_i, left_r7_c3_i, left_r7_c4_i, left_r7_c5_i, left_r7_c6_i, left_r7_c7_i, left_r7_c8_i, left_r7_c9_i, left_r7_c10_i, left_r7_c11_i, left_r7_c12_i, left_r7_c13_i, left_r7_c14_i, left_r7_c15_i, left_r7_c16_i, left_r7_c17_i, left_r7_c18_i;
    __m256 left_r8_c0_r, left_r8_c1_r, left_r8_c2_r, left_r8_c3_r, left_r8_c4_r, left_r8_c5_r, left_r8_c6_r, left_r8_c7_r, left_r8_c8_r, left_r8_c9_r, left_r8_c10_r, left_r8_c11_r, left_r8_c12_r, left_r8_c13_r, left_r8_c14_r, left_r8_c15_r, left_r8_c16_r, left_r8_c17_r, left_r8_c18_r;
    __m256 left_r8_c0_i, left_r8_c1_i, left_r8_c2_i, left_r8_c3_i, left_r8_c4_i, left_r8_c5_i, left_r8_c6_i, left_r8_c7_i, left_r8_c8_i, left_r8_c9_i, left_r8_c10_i, left_r8_c11_i, left_r8_c12_i, left_r8_c13_i, left_r8_c14_i, left_r8_c15_i, left_r8_c16_i, left_r8_c17_i, left_r8_c18_i;
    __m256 left_r9_c0_r, left_r9_c1_r, left_r9_c2_r, left_r9_c3_r, left_r9_c4_r, left_r9_c5_r, left_r9_c6_r, left_r9_c7_r, left_r9_c8_r, left_r9_c9_r, left_r9_c10_r, left_r9_c11_r, left_r9_c12_r, left_r9_c13_r, left_r9_c14_r, left_r9_c15_r, left_r9_c16_r, left_r9_c17_r, left_r9_c18_r;
    __m256 left_r9_c0_i, left_r9_c1_i, left_r9_c2_i, left_r9_c3_i, left_r9_c4_i, left_r9_c5_i, left_r9_c6_i, left_r9_c7_i, left_r9_c8_i, left_r9_c9_i, left_r9_c10_i, left_r9_c11_i, left_r9_c12_i, left_r9_c13_i, left_r9_c14_i, left_r9_c15_i, left_r9_c16_i, left_r9_c17_i, left_r9_c18_i;
    __m256 left_r10_c0_r, left_r10_c1_r, left_r10_c2_r, left_r10_c3_r, left_r10_c4_r, left_r10_c5_r, left_r10_c6_r, left_r10_c7_r, left_r10_c8_r, left_r10_c9_r, left_r10_c10_r, left_r10_c11_r, left_r10_c12_r, left_r10_c13_r, left_r10_c14_r, left_r10_c15_r, left_r10_c16_r, left_r10_c17_r, left_r10_c18_r;
    __m256 left_r10_c0_i, left_r10_c1_i, left_r10_c2_i, left_r10_c3_i, left_r10_c4_i, left_r10_c5_i, left_r10_c6_i, left_r10_c7_i, left_r10_c8_i, left_r10_c9_i, left_r10_c10_i, left_r10_c11_i, left_r10_c12_i, left_r10_c13_i, left_r10_c14_i, left_r10_c15_i, left_r10_c16_i, left_r10_c17_i, left_r10_c18_i;
    __m256 left_r11_c0_r, left_r11_c1_r, left_r11_c2_r, left_r11_c3_r, left_r11_c4_r, left_r11_c5_r, left_r11_c6_r, left_r11_c7_r, left_r11_c8_r, left_r11_c9_r, left_r11_c10_r, left_r11_c11_r, left_r11_c12_r, left_r11_c13_r, left_r11_c14_r, left_r11_c15_r, left_r11_c16_r, left_r11_c17_r, left_r11_c18_r;
    __m256 left_r11_c0_i, left_r11_c1_i, left_r11_c2_i, left_r11_c3_i, left_r11_c4_i, left_r11_c5_i, left_r11_c6_i, left_r11_c7_i, left_r11_c8_i, left_r11_c9_i, left_r11_c10_i, left_r11_c11_i, left_r11_c12_i, left_r11_c13_i, left_r11_c14_i, left_r11_c15_i, left_r11_c16_i, left_r11_c17_i, left_r11_c18_i;
    __m256 left_r12_c0_r, left_r12_c1_r, left_r12_c2_r, left_r12_c3_r, left_r12_c4_r, left_r12_c5_r, left_r12_c6_r, left_r12_c7_r, left_r12_c8_r, left_r12_c9_r, left_r12_c10_r, left_r12_c11_r, left_r12_c12_r, left_r12_c13_r, left_r12_c14_r, left_r12_c15_r, left_r12_c16_r, left_r12_c17_r, left_r12_c18_r;
    __m256 left_r12_c0_i, left_r12_c1_i, left_r12_c2_i, left_r12_c3_i, left_r12_c4_i, left_r12_c5_i, left_r12_c6_i, left_r12_c7_i, left_r12_c8_i, left_r12_c9_i, left_r12_c10_i, left_r12_c11_i, left_r12_c12_i, left_r12_c13_i, left_r12_c14_i, left_r12_c15_i, left_r12_c16_i, left_r12_c17_i, left_r12_c18_i;
    __m256 left_r13_c0_r, left_r13_c1_r, left_r13_c2_r, left_r13_c3_r, left_r13_c4_r, left_r13_c5_r, left_r13_c6_r, left_r13_c7_r, left_r13_c8_r, left_r13_c9_r, left_r13_c10_r, left_r13_c11_r, left_r13_c12_r, left_r13_c13_r, left_r13_c14_r, left_r13_c15_r, left_r13_c16_r, left_r13_c17_r, left_r13_c18_r;
    __m256 left_r13_c0_i, left_r13_c1_i, left_r13_c2_i, left_r13_c3_i, left_r13_c4_i, left_r13_c5_i, left_r13_c6_i, left_r13_c7_i, left_r13_c8_i, left_r13_c9_i, left_r13_c10_i, left_r13_c11_i, left_r13_c12_i, left_r13_c13_i, left_r13_c14_i, left_r13_c15_i, left_r13_c16_i, left_r13_c17_i, left_r13_c18_i;
    __m256 left_r14_c0_r, left_r14_c1_r, left_r14_c2_r, left_r14_c3_r, left_r14_c4_r, left_r14_c5_r, left_r14_c6_r, left_r14_c7_r, left_r14_c8_r, left_r14_c9_r, left_r14_c10_r, left_r14_c11_r, left_r14_c12_r, left_r14_c13_r, left_r14_c14_r, left_r14_c15_r, left_r14_c16_r, left_r14_c17_r, left_r14_c18_r;
    __m256 left_r14_c0_i, left_r14_c1_i, left_r14_c2_i, left_r14_c3_i, left_r14_c4_i, left_r14_c5_i, left_r14_c6_i, left_r14_c7_i, left_r14_c8_i, left_r14_c9_i, left_r14_c10_i, left_r14_c11_i, left_r14_c12_i, left_r14_c13_i, left_r14_c14_i, left_r14_c15_i, left_r14_c16_i, left_r14_c17_i, left_r14_c18_i;
    __m256 left_r15_c0_r, left_r15_c1_r, left_r15_c2_r, left_r15_c3_r, left_r15_c4_r, left_r15_c5_r, left_r15_c6_r, left_r15_c7_r, left_r15_c8_r, left_r15_c9_r, left_r15_c10_r, left_r15_c11_r, left_r15_c12_r, left_r15_c13_r, left_r15_c14_r, left_r15_c15_r, left_r15_c16_r, left_r15_c17_r, left_r15_c18_r;
    __m256 left_r15_c0_i, left_r15_c1_i, left_r15_c2_i, left_r15_c3_i, left_r15_c4_i, left_r15_c5_i, left_r15_c6_i, left_r15_c7_i, left_r15_c8_i, left_r15_c9_i, left_r15_c10_i, left_r15_c11_i, left_r15_c12_i, left_r15_c13_i, left_r15_c14_i, left_r15_c15_i, left_r15_c16_i, left_r15_c17_i, left_r15_c18_i;
    __m256 left_r16_c0_r, left_r16_c1_r, left_r16_c2_r, left_r16_c3_r, left_r16_c4_r, left_r16_c5_r, left_r16_c6_r, left_r16_c7_r, left_r16_c8_r, left_r16_c9_r, left_r16_c10_r, left_r16_c11_r, left_r16_c12_r, left_r16_c13_r, left_r16_c14_r, left_r16_c15_r, left_r16_c16_r, left_r16_c17_r, left_r16_c18_r;
    __m256 left_r16_c0_i, left_r16_c1_i, left_r16_c2_i, left_r16_c3_i, left_r16_c4_i, left_r16_c5_i, left_r16_c6_i, left_r16_c7_i, left_r16_c8_i, left_r16_c9_i, left_r16_c10_i, left_r16_c11_i, left_r16_c12_i, left_r16_c13_i, left_r16_c14_i, left_r16_c15_i, left_r16_c16_i, left_r16_c17_i, left_r16_c18_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    __m256 output_r3;
    __m256 output_r4;
    __m256 output_r5;
    __m256 output_r6;
    __m256 output_r7;
    __m256 output_r8;
    __m256 output_r9;
    __m256 output_r10;
    __m256 output_r11;
    __m256 output_r12;
    __m256 output_r13;
    __m256 output_r14;
    __m256 output_r15;
    __m256 output_r16;
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
    left_r0_c4_r = _mm256_broadcast_ss (&(A[0] + 4)->real);
    left_r0_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 4)->imag));
    left_r0_c5_r = _mm256_broadcast_ss (&(A[0] + 5)->real);
    left_r0_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 5)->imag));
    left_r0_c6_r = _mm256_broadcast_ss (&(A[0] + 6)->real);
    left_r0_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 6)->imag));
    left_r0_c7_r = _mm256_broadcast_ss (&(A[0] + 7)->real);
    left_r0_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 7)->imag));
    left_r0_c8_r = _mm256_broadcast_ss (&(A[0] + 8)->real);
    left_r0_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 8)->imag));
    left_r0_c9_r = _mm256_broadcast_ss (&(A[0] + 9)->real);
    left_r0_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 9)->imag));
    left_r0_c10_r = _mm256_broadcast_ss (&(A[0] + 10)->real);
    left_r0_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 10)->imag));
    left_r0_c11_r = _mm256_broadcast_ss (&(A[0] + 11)->real);
    left_r0_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 11)->imag));
    left_r0_c12_r = _mm256_broadcast_ss (&(A[0] + 12)->real);
    left_r0_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 12)->imag));
    left_r0_c13_r = _mm256_broadcast_ss (&(A[0] + 13)->real);
    left_r0_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 13)->imag));
    left_r0_c14_r = _mm256_broadcast_ss (&(A[0] + 14)->real);
    left_r0_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 14)->imag));
    left_r0_c15_r = _mm256_broadcast_ss (&(A[0] + 15)->real);
    left_r0_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 15)->imag));
    left_r0_c16_r = _mm256_broadcast_ss (&(A[0] + 16)->real);
    left_r0_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 16)->imag));
    left_r0_c17_r = _mm256_broadcast_ss (&(A[0] + 17)->real);
    left_r0_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 17)->imag));
    left_r0_c18_r = _mm256_broadcast_ss (&(A[0] + 18)->real);
    left_r0_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[0] + 18)->imag));

    left_r1_c0_r = _mm256_broadcast_ss (&(A[1] + 0)->real);
    left_r1_c0_i = _mm256_broadcast_ss (&(A[1] + 0)->imag);
    left_r1_c1_r = _mm256_broadcast_ss (&(A[1] + 1)->real);
    left_r1_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 1)->imag));
    left_r1_c2_r = _mm256_broadcast_ss (&(A[1] + 2)->real);
    left_r1_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 2)->imag));
    left_r1_c3_r = _mm256_broadcast_ss (&(A[1] + 3)->real);
    left_r1_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 3)->imag));
    left_r1_c4_r = _mm256_broadcast_ss (&(A[1] + 4)->real);
    left_r1_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 4)->imag));
    left_r1_c5_r = _mm256_broadcast_ss (&(A[1] + 5)->real);
    left_r1_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 5)->imag));
    left_r1_c6_r = _mm256_broadcast_ss (&(A[1] + 6)->real);
    left_r1_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 6)->imag));
    left_r1_c7_r = _mm256_broadcast_ss (&(A[1] + 7)->real);
    left_r1_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 7)->imag));
    left_r1_c8_r = _mm256_broadcast_ss (&(A[1] + 8)->real);
    left_r1_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 8)->imag));
    left_r1_c9_r = _mm256_broadcast_ss (&(A[1] + 9)->real);
    left_r1_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 9)->imag));
    left_r1_c10_r = _mm256_broadcast_ss (&(A[1] + 10)->real);
    left_r1_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 10)->imag));
    left_r1_c11_r = _mm256_broadcast_ss (&(A[1] + 11)->real);
    left_r1_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 11)->imag));
    left_r1_c12_r = _mm256_broadcast_ss (&(A[1] + 12)->real);
    left_r1_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 12)->imag));
    left_r1_c13_r = _mm256_broadcast_ss (&(A[1] + 13)->real);
    left_r1_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 13)->imag));
    left_r1_c14_r = _mm256_broadcast_ss (&(A[1] + 14)->real);
    left_r1_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 14)->imag));
    left_r1_c15_r = _mm256_broadcast_ss (&(A[1] + 15)->real);
    left_r1_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 15)->imag));
    left_r1_c16_r = _mm256_broadcast_ss (&(A[1] + 16)->real);
    left_r1_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 16)->imag));
    left_r1_c17_r = _mm256_broadcast_ss (&(A[1] + 17)->real);
    left_r1_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 17)->imag));
    left_r1_c18_r = _mm256_broadcast_ss (&(A[1] + 18)->real);
    left_r1_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[1] + 18)->imag));

    left_r2_c0_r = _mm256_broadcast_ss (&(A[2] + 0)->real);
    left_r2_c0_i = _mm256_broadcast_ss (&(A[2] + 0)->imag);
    left_r2_c1_r = _mm256_broadcast_ss (&(A[2] + 1)->real);
    left_r2_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 1)->imag));
    left_r2_c2_r = _mm256_broadcast_ss (&(A[2] + 2)->real);
    left_r2_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 2)->imag));
    left_r2_c3_r = _mm256_broadcast_ss (&(A[2] + 3)->real);
    left_r2_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 3)->imag));
    left_r2_c4_r = _mm256_broadcast_ss (&(A[2] + 4)->real);
    left_r2_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 4)->imag));
    left_r2_c5_r = _mm256_broadcast_ss (&(A[2] + 5)->real);
    left_r2_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 5)->imag));
    left_r2_c6_r = _mm256_broadcast_ss (&(A[2] + 6)->real);
    left_r2_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 6)->imag));
    left_r2_c7_r = _mm256_broadcast_ss (&(A[2] + 7)->real);
    left_r2_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 7)->imag));
    left_r2_c8_r = _mm256_broadcast_ss (&(A[2] + 8)->real);
    left_r2_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 8)->imag));
    left_r2_c9_r = _mm256_broadcast_ss (&(A[2] + 9)->real);
    left_r2_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 9)->imag));
    left_r2_c10_r = _mm256_broadcast_ss (&(A[2] + 10)->real);
    left_r2_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 10)->imag));
    left_r2_c11_r = _mm256_broadcast_ss (&(A[2] + 11)->real);
    left_r2_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 11)->imag));
    left_r2_c12_r = _mm256_broadcast_ss (&(A[2] + 12)->real);
    left_r2_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 12)->imag));
    left_r2_c13_r = _mm256_broadcast_ss (&(A[2] + 13)->real);
    left_r2_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 13)->imag));
    left_r2_c14_r = _mm256_broadcast_ss (&(A[2] + 14)->real);
    left_r2_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 14)->imag));
    left_r2_c15_r = _mm256_broadcast_ss (&(A[2] + 15)->real);
    left_r2_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 15)->imag));
    left_r2_c16_r = _mm256_broadcast_ss (&(A[2] + 16)->real);
    left_r2_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 16)->imag));
    left_r2_c17_r = _mm256_broadcast_ss (&(A[2] + 17)->real);
    left_r2_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 17)->imag));
    left_r2_c18_r = _mm256_broadcast_ss (&(A[2] + 18)->real);
    left_r2_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[2] + 18)->imag));

    left_r3_c0_r = _mm256_broadcast_ss (&(A[3] + 0)->real);
    left_r3_c0_i = _mm256_broadcast_ss (&(A[3] + 0)->imag);
    left_r3_c1_r = _mm256_broadcast_ss (&(A[3] + 1)->real);
    left_r3_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 1)->imag));
    left_r3_c2_r = _mm256_broadcast_ss (&(A[3] + 2)->real);
    left_r3_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 2)->imag));
    left_r3_c3_r = _mm256_broadcast_ss (&(A[3] + 3)->real);
    left_r3_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 3)->imag));
    left_r3_c4_r = _mm256_broadcast_ss (&(A[3] + 4)->real);
    left_r3_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 4)->imag));
    left_r3_c5_r = _mm256_broadcast_ss (&(A[3] + 5)->real);
    left_r3_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 5)->imag));
    left_r3_c6_r = _mm256_broadcast_ss (&(A[3] + 6)->real);
    left_r3_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 6)->imag));
    left_r3_c7_r = _mm256_broadcast_ss (&(A[3] + 7)->real);
    left_r3_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 7)->imag));
    left_r3_c8_r = _mm256_broadcast_ss (&(A[3] + 8)->real);
    left_r3_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 8)->imag));
    left_r3_c9_r = _mm256_broadcast_ss (&(A[3] + 9)->real);
    left_r3_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 9)->imag));
    left_r3_c10_r = _mm256_broadcast_ss (&(A[3] + 10)->real);
    left_r3_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 10)->imag));
    left_r3_c11_r = _mm256_broadcast_ss (&(A[3] + 11)->real);
    left_r3_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 11)->imag));
    left_r3_c12_r = _mm256_broadcast_ss (&(A[3] + 12)->real);
    left_r3_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 12)->imag));
    left_r3_c13_r = _mm256_broadcast_ss (&(A[3] + 13)->real);
    left_r3_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 13)->imag));
    left_r3_c14_r = _mm256_broadcast_ss (&(A[3] + 14)->real);
    left_r3_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 14)->imag));
    left_r3_c15_r = _mm256_broadcast_ss (&(A[3] + 15)->real);
    left_r3_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 15)->imag));
    left_r3_c16_r = _mm256_broadcast_ss (&(A[3] + 16)->real);
    left_r3_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 16)->imag));
    left_r3_c17_r = _mm256_broadcast_ss (&(A[3] + 17)->real);
    left_r3_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 17)->imag));
    left_r3_c18_r = _mm256_broadcast_ss (&(A[3] + 18)->real);
    left_r3_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[3] + 18)->imag));

    left_r4_c0_r = _mm256_broadcast_ss (&(A[4] + 0)->real);
    left_r4_c0_i = _mm256_broadcast_ss (&(A[4] + 0)->imag);
    left_r4_c1_r = _mm256_broadcast_ss (&(A[4] + 1)->real);
    left_r4_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 1)->imag));
    left_r4_c2_r = _mm256_broadcast_ss (&(A[4] + 2)->real);
    left_r4_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 2)->imag));
    left_r4_c3_r = _mm256_broadcast_ss (&(A[4] + 3)->real);
    left_r4_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 3)->imag));
    left_r4_c4_r = _mm256_broadcast_ss (&(A[4] + 4)->real);
    left_r4_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 4)->imag));
    left_r4_c5_r = _mm256_broadcast_ss (&(A[4] + 5)->real);
    left_r4_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 5)->imag));
    left_r4_c6_r = _mm256_broadcast_ss (&(A[4] + 6)->real);
    left_r4_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 6)->imag));
    left_r4_c7_r = _mm256_broadcast_ss (&(A[4] + 7)->real);
    left_r4_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 7)->imag));
    left_r4_c8_r = _mm256_broadcast_ss (&(A[4] + 8)->real);
    left_r4_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 8)->imag));
    left_r4_c9_r = _mm256_broadcast_ss (&(A[4] + 9)->real);
    left_r4_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 9)->imag));
    left_r4_c10_r = _mm256_broadcast_ss (&(A[4] + 10)->real);
    left_r4_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 10)->imag));
    left_r4_c11_r = _mm256_broadcast_ss (&(A[4] + 11)->real);
    left_r4_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 11)->imag));
    left_r4_c12_r = _mm256_broadcast_ss (&(A[4] + 12)->real);
    left_r4_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 12)->imag));
    left_r4_c13_r = _mm256_broadcast_ss (&(A[4] + 13)->real);
    left_r4_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 13)->imag));
    left_r4_c14_r = _mm256_broadcast_ss (&(A[4] + 14)->real);
    left_r4_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 14)->imag));
    left_r4_c15_r = _mm256_broadcast_ss (&(A[4] + 15)->real);
    left_r4_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 15)->imag));
    left_r4_c16_r = _mm256_broadcast_ss (&(A[4] + 16)->real);
    left_r4_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 16)->imag));
    left_r4_c17_r = _mm256_broadcast_ss (&(A[4] + 17)->real);
    left_r4_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 17)->imag));
    left_r4_c18_r = _mm256_broadcast_ss (&(A[4] + 18)->real);
    left_r4_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[4] + 18)->imag));

    left_r5_c0_r = _mm256_broadcast_ss (&(A[5] + 0)->real);
    left_r5_c0_i = _mm256_broadcast_ss (&(A[5] + 0)->imag);
    left_r5_c1_r = _mm256_broadcast_ss (&(A[5] + 1)->real);
    left_r5_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 1)->imag));
    left_r5_c2_r = _mm256_broadcast_ss (&(A[5] + 2)->real);
    left_r5_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 2)->imag));
    left_r5_c3_r = _mm256_broadcast_ss (&(A[5] + 3)->real);
    left_r5_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 3)->imag));
    left_r5_c4_r = _mm256_broadcast_ss (&(A[5] + 4)->real);
    left_r5_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 4)->imag));
    left_r5_c5_r = _mm256_broadcast_ss (&(A[5] + 5)->real);
    left_r5_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 5)->imag));
    left_r5_c6_r = _mm256_broadcast_ss (&(A[5] + 6)->real);
    left_r5_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 6)->imag));
    left_r5_c7_r = _mm256_broadcast_ss (&(A[5] + 7)->real);
    left_r5_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 7)->imag));
    left_r5_c8_r = _mm256_broadcast_ss (&(A[5] + 8)->real);
    left_r5_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 8)->imag));
    left_r5_c9_r = _mm256_broadcast_ss (&(A[5] + 9)->real);
    left_r5_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 9)->imag));
    left_r5_c10_r = _mm256_broadcast_ss (&(A[5] + 10)->real);
    left_r5_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 10)->imag));
    left_r5_c11_r = _mm256_broadcast_ss (&(A[5] + 11)->real);
    left_r5_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 11)->imag));
    left_r5_c12_r = _mm256_broadcast_ss (&(A[5] + 12)->real);
    left_r5_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 12)->imag));
    left_r5_c13_r = _mm256_broadcast_ss (&(A[5] + 13)->real);
    left_r5_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 13)->imag));
    left_r5_c14_r = _mm256_broadcast_ss (&(A[5] + 14)->real);
    left_r5_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 14)->imag));
    left_r5_c15_r = _mm256_broadcast_ss (&(A[5] + 15)->real);
    left_r5_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 15)->imag));
    left_r5_c16_r = _mm256_broadcast_ss (&(A[5] + 16)->real);
    left_r5_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 16)->imag));
    left_r5_c17_r = _mm256_broadcast_ss (&(A[5] + 17)->real);
    left_r5_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 17)->imag));
    left_r5_c18_r = _mm256_broadcast_ss (&(A[5] + 18)->real);
    left_r5_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[5] + 18)->imag));

    left_r6_c0_r = _mm256_broadcast_ss (&(A[6] + 0)->real);
    left_r6_c0_i = _mm256_broadcast_ss (&(A[6] + 0)->imag);
    left_r6_c1_r = _mm256_broadcast_ss (&(A[6] + 1)->real);
    left_r6_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 1)->imag));
    left_r6_c2_r = _mm256_broadcast_ss (&(A[6] + 2)->real);
    left_r6_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 2)->imag));
    left_r6_c3_r = _mm256_broadcast_ss (&(A[6] + 3)->real);
    left_r6_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 3)->imag));
    left_r6_c4_r = _mm256_broadcast_ss (&(A[6] + 4)->real);
    left_r6_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 4)->imag));
    left_r6_c5_r = _mm256_broadcast_ss (&(A[6] + 5)->real);
    left_r6_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 5)->imag));
    left_r6_c6_r = _mm256_broadcast_ss (&(A[6] + 6)->real);
    left_r6_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 6)->imag));
    left_r6_c7_r = _mm256_broadcast_ss (&(A[6] + 7)->real);
    left_r6_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 7)->imag));
    left_r6_c8_r = _mm256_broadcast_ss (&(A[6] + 8)->real);
    left_r6_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 8)->imag));
    left_r6_c9_r = _mm256_broadcast_ss (&(A[6] + 9)->real);
    left_r6_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 9)->imag));
    left_r6_c10_r = _mm256_broadcast_ss (&(A[6] + 10)->real);
    left_r6_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 10)->imag));
    left_r6_c11_r = _mm256_broadcast_ss (&(A[6] + 11)->real);
    left_r6_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 11)->imag));
    left_r6_c12_r = _mm256_broadcast_ss (&(A[6] + 12)->real);
    left_r6_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 12)->imag));
    left_r6_c13_r = _mm256_broadcast_ss (&(A[6] + 13)->real);
    left_r6_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 13)->imag));
    left_r6_c14_r = _mm256_broadcast_ss (&(A[6] + 14)->real);
    left_r6_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 14)->imag));
    left_r6_c15_r = _mm256_broadcast_ss (&(A[6] + 15)->real);
    left_r6_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 15)->imag));
    left_r6_c16_r = _mm256_broadcast_ss (&(A[6] + 16)->real);
    left_r6_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 16)->imag));
    left_r6_c17_r = _mm256_broadcast_ss (&(A[6] + 17)->real);
    left_r6_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 17)->imag));
    left_r6_c18_r = _mm256_broadcast_ss (&(A[6] + 18)->real);
    left_r6_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[6] + 18)->imag));

    left_r7_c0_r = _mm256_broadcast_ss (&(A[7] + 0)->real);
    left_r7_c0_i = _mm256_broadcast_ss (&(A[7] + 0)->imag);
    left_r7_c1_r = _mm256_broadcast_ss (&(A[7] + 1)->real);
    left_r7_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 1)->imag));
    left_r7_c2_r = _mm256_broadcast_ss (&(A[7] + 2)->real);
    left_r7_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 2)->imag));
    left_r7_c3_r = _mm256_broadcast_ss (&(A[7] + 3)->real);
    left_r7_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 3)->imag));
    left_r7_c4_r = _mm256_broadcast_ss (&(A[7] + 4)->real);
    left_r7_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 4)->imag));
    left_r7_c5_r = _mm256_broadcast_ss (&(A[7] + 5)->real);
    left_r7_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 5)->imag));
    left_r7_c6_r = _mm256_broadcast_ss (&(A[7] + 6)->real);
    left_r7_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 6)->imag));
    left_r7_c7_r = _mm256_broadcast_ss (&(A[7] + 7)->real);
    left_r7_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 7)->imag));
    left_r7_c8_r = _mm256_broadcast_ss (&(A[7] + 8)->real);
    left_r7_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 8)->imag));
    left_r7_c9_r = _mm256_broadcast_ss (&(A[7] + 9)->real);
    left_r7_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 9)->imag));
    left_r7_c10_r = _mm256_broadcast_ss (&(A[7] + 10)->real);
    left_r7_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 10)->imag));
    left_r7_c11_r = _mm256_broadcast_ss (&(A[7] + 11)->real);
    left_r7_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 11)->imag));
    left_r7_c12_r = _mm256_broadcast_ss (&(A[7] + 12)->real);
    left_r7_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 12)->imag));
    left_r7_c13_r = _mm256_broadcast_ss (&(A[7] + 13)->real);
    left_r7_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 13)->imag));
    left_r7_c14_r = _mm256_broadcast_ss (&(A[7] + 14)->real);
    left_r7_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 14)->imag));
    left_r7_c15_r = _mm256_broadcast_ss (&(A[7] + 15)->real);
    left_r7_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 15)->imag));
    left_r7_c16_r = _mm256_broadcast_ss (&(A[7] + 16)->real);
    left_r7_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 16)->imag));
    left_r7_c17_r = _mm256_broadcast_ss (&(A[7] + 17)->real);
    left_r7_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 17)->imag));
    left_r7_c18_r = _mm256_broadcast_ss (&(A[7] + 18)->real);
    left_r7_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[7] + 18)->imag));

    left_r8_c0_r = _mm256_broadcast_ss (&(A[8] + 0)->real);
    left_r8_c0_i = _mm256_broadcast_ss (&(A[8] + 0)->imag);
    left_r8_c1_r = _mm256_broadcast_ss (&(A[8] + 1)->real);
    left_r8_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 1)->imag));
    left_r8_c2_r = _mm256_broadcast_ss (&(A[8] + 2)->real);
    left_r8_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 2)->imag));
    left_r8_c3_r = _mm256_broadcast_ss (&(A[8] + 3)->real);
    left_r8_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 3)->imag));
    left_r8_c4_r = _mm256_broadcast_ss (&(A[8] + 4)->real);
    left_r8_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 4)->imag));
    left_r8_c5_r = _mm256_broadcast_ss (&(A[8] + 5)->real);
    left_r8_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 5)->imag));
    left_r8_c6_r = _mm256_broadcast_ss (&(A[8] + 6)->real);
    left_r8_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 6)->imag));
    left_r8_c7_r = _mm256_broadcast_ss (&(A[8] + 7)->real);
    left_r8_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 7)->imag));
    left_r8_c8_r = _mm256_broadcast_ss (&(A[8] + 8)->real);
    left_r8_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 8)->imag));
    left_r8_c9_r = _mm256_broadcast_ss (&(A[8] + 9)->real);
    left_r8_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 9)->imag));
    left_r8_c10_r = _mm256_broadcast_ss (&(A[8] + 10)->real);
    left_r8_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 10)->imag));
    left_r8_c11_r = _mm256_broadcast_ss (&(A[8] + 11)->real);
    left_r8_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 11)->imag));
    left_r8_c12_r = _mm256_broadcast_ss (&(A[8] + 12)->real);
    left_r8_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 12)->imag));
    left_r8_c13_r = _mm256_broadcast_ss (&(A[8] + 13)->real);
    left_r8_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 13)->imag));
    left_r8_c14_r = _mm256_broadcast_ss (&(A[8] + 14)->real);
    left_r8_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 14)->imag));
    left_r8_c15_r = _mm256_broadcast_ss (&(A[8] + 15)->real);
    left_r8_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 15)->imag));
    left_r8_c16_r = _mm256_broadcast_ss (&(A[8] + 16)->real);
    left_r8_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 16)->imag));
    left_r8_c17_r = _mm256_broadcast_ss (&(A[8] + 17)->real);
    left_r8_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 17)->imag));
    left_r8_c18_r = _mm256_broadcast_ss (&(A[8] + 18)->real);
    left_r8_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[8] + 18)->imag));

    left_r9_c0_r = _mm256_broadcast_ss (&(A[9] + 0)->real);
    left_r9_c0_i = _mm256_broadcast_ss (&(A[9] + 0)->imag);
    left_r9_c1_r = _mm256_broadcast_ss (&(A[9] + 1)->real);
    left_r9_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 1)->imag));
    left_r9_c2_r = _mm256_broadcast_ss (&(A[9] + 2)->real);
    left_r9_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 2)->imag));
    left_r9_c3_r = _mm256_broadcast_ss (&(A[9] + 3)->real);
    left_r9_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 3)->imag));
    left_r9_c4_r = _mm256_broadcast_ss (&(A[9] + 4)->real);
    left_r9_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 4)->imag));
    left_r9_c5_r = _mm256_broadcast_ss (&(A[9] + 5)->real);
    left_r9_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 5)->imag));
    left_r9_c6_r = _mm256_broadcast_ss (&(A[9] + 6)->real);
    left_r9_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 6)->imag));
    left_r9_c7_r = _mm256_broadcast_ss (&(A[9] + 7)->real);
    left_r9_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 7)->imag));
    left_r9_c8_r = _mm256_broadcast_ss (&(A[9] + 8)->real);
    left_r9_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 8)->imag));
    left_r9_c9_r = _mm256_broadcast_ss (&(A[9] + 9)->real);
    left_r9_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 9)->imag));
    left_r9_c10_r = _mm256_broadcast_ss (&(A[9] + 10)->real);
    left_r9_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 10)->imag));
    left_r9_c11_r = _mm256_broadcast_ss (&(A[9] + 11)->real);
    left_r9_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 11)->imag));
    left_r9_c12_r = _mm256_broadcast_ss (&(A[9] + 12)->real);
    left_r9_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 12)->imag));
    left_r9_c13_r = _mm256_broadcast_ss (&(A[9] + 13)->real);
    left_r9_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 13)->imag));
    left_r9_c14_r = _mm256_broadcast_ss (&(A[9] + 14)->real);
    left_r9_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 14)->imag));
    left_r9_c15_r = _mm256_broadcast_ss (&(A[9] + 15)->real);
    left_r9_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 15)->imag));
    left_r9_c16_r = _mm256_broadcast_ss (&(A[9] + 16)->real);
    left_r9_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 16)->imag));
    left_r9_c17_r = _mm256_broadcast_ss (&(A[9] + 17)->real);
    left_r9_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 17)->imag));
    left_r9_c18_r = _mm256_broadcast_ss (&(A[9] + 18)->real);
    left_r9_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[9] + 18)->imag));

    left_r10_c0_r = _mm256_broadcast_ss (&(A[10] + 0)->real);
    left_r10_c0_i = _mm256_broadcast_ss (&(A[10] + 0)->imag);
    left_r10_c1_r = _mm256_broadcast_ss (&(A[10] + 1)->real);
    left_r10_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 1)->imag));
    left_r10_c2_r = _mm256_broadcast_ss (&(A[10] + 2)->real);
    left_r10_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 2)->imag));
    left_r10_c3_r = _mm256_broadcast_ss (&(A[10] + 3)->real);
    left_r10_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 3)->imag));
    left_r10_c4_r = _mm256_broadcast_ss (&(A[10] + 4)->real);
    left_r10_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 4)->imag));
    left_r10_c5_r = _mm256_broadcast_ss (&(A[10] + 5)->real);
    left_r10_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 5)->imag));
    left_r10_c6_r = _mm256_broadcast_ss (&(A[10] + 6)->real);
    left_r10_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 6)->imag));
    left_r10_c7_r = _mm256_broadcast_ss (&(A[10] + 7)->real);
    left_r10_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 7)->imag));
    left_r10_c8_r = _mm256_broadcast_ss (&(A[10] + 8)->real);
    left_r10_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 8)->imag));
    left_r10_c9_r = _mm256_broadcast_ss (&(A[10] + 9)->real);
    left_r10_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 9)->imag));
    left_r10_c10_r = _mm256_broadcast_ss (&(A[10] + 10)->real);
    left_r10_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 10)->imag));
    left_r10_c11_r = _mm256_broadcast_ss (&(A[10] + 11)->real);
    left_r10_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 11)->imag));
    left_r10_c12_r = _mm256_broadcast_ss (&(A[10] + 12)->real);
    left_r10_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 12)->imag));
    left_r10_c13_r = _mm256_broadcast_ss (&(A[10] + 13)->real);
    left_r10_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 13)->imag));
    left_r10_c14_r = _mm256_broadcast_ss (&(A[10] + 14)->real);
    left_r10_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 14)->imag));
    left_r10_c15_r = _mm256_broadcast_ss (&(A[10] + 15)->real);
    left_r10_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 15)->imag));
    left_r10_c16_r = _mm256_broadcast_ss (&(A[10] + 16)->real);
    left_r10_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 16)->imag));
    left_r10_c17_r = _mm256_broadcast_ss (&(A[10] + 17)->real);
    left_r10_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 17)->imag));
    left_r10_c18_r = _mm256_broadcast_ss (&(A[10] + 18)->real);
    left_r10_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[10] + 18)->imag));

    left_r11_c0_r = _mm256_broadcast_ss (&(A[11] + 0)->real);
    left_r11_c0_i = _mm256_broadcast_ss (&(A[11] + 0)->imag);
    left_r11_c1_r = _mm256_broadcast_ss (&(A[11] + 1)->real);
    left_r11_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 1)->imag));
    left_r11_c2_r = _mm256_broadcast_ss (&(A[11] + 2)->real);
    left_r11_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 2)->imag));
    left_r11_c3_r = _mm256_broadcast_ss (&(A[11] + 3)->real);
    left_r11_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 3)->imag));
    left_r11_c4_r = _mm256_broadcast_ss (&(A[11] + 4)->real);
    left_r11_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 4)->imag));
    left_r11_c5_r = _mm256_broadcast_ss (&(A[11] + 5)->real);
    left_r11_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 5)->imag));
    left_r11_c6_r = _mm256_broadcast_ss (&(A[11] + 6)->real);
    left_r11_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 6)->imag));
    left_r11_c7_r = _mm256_broadcast_ss (&(A[11] + 7)->real);
    left_r11_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 7)->imag));
    left_r11_c8_r = _mm256_broadcast_ss (&(A[11] + 8)->real);
    left_r11_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 8)->imag));
    left_r11_c9_r = _mm256_broadcast_ss (&(A[11] + 9)->real);
    left_r11_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 9)->imag));
    left_r11_c10_r = _mm256_broadcast_ss (&(A[11] + 10)->real);
    left_r11_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 10)->imag));
    left_r11_c11_r = _mm256_broadcast_ss (&(A[11] + 11)->real);
    left_r11_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 11)->imag));
    left_r11_c12_r = _mm256_broadcast_ss (&(A[11] + 12)->real);
    left_r11_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 12)->imag));
    left_r11_c13_r = _mm256_broadcast_ss (&(A[11] + 13)->real);
    left_r11_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 13)->imag));
    left_r11_c14_r = _mm256_broadcast_ss (&(A[11] + 14)->real);
    left_r11_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 14)->imag));
    left_r11_c15_r = _mm256_broadcast_ss (&(A[11] + 15)->real);
    left_r11_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 15)->imag));
    left_r11_c16_r = _mm256_broadcast_ss (&(A[11] + 16)->real);
    left_r11_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 16)->imag));
    left_r11_c17_r = _mm256_broadcast_ss (&(A[11] + 17)->real);
    left_r11_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 17)->imag));
    left_r11_c18_r = _mm256_broadcast_ss (&(A[11] + 18)->real);
    left_r11_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[11] + 18)->imag));

    left_r12_c0_r = _mm256_broadcast_ss (&(A[12] + 0)->real);
    left_r12_c0_i = _mm256_broadcast_ss (&(A[12] + 0)->imag);
    left_r12_c1_r = _mm256_broadcast_ss (&(A[12] + 1)->real);
    left_r12_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 1)->imag));
    left_r12_c2_r = _mm256_broadcast_ss (&(A[12] + 2)->real);
    left_r12_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 2)->imag));
    left_r12_c3_r = _mm256_broadcast_ss (&(A[12] + 3)->real);
    left_r12_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 3)->imag));
    left_r12_c4_r = _mm256_broadcast_ss (&(A[12] + 4)->real);
    left_r12_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 4)->imag));
    left_r12_c5_r = _mm256_broadcast_ss (&(A[12] + 5)->real);
    left_r12_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 5)->imag));
    left_r12_c6_r = _mm256_broadcast_ss (&(A[12] + 6)->real);
    left_r12_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 6)->imag));
    left_r12_c7_r = _mm256_broadcast_ss (&(A[12] + 7)->real);
    left_r12_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 7)->imag));
    left_r12_c8_r = _mm256_broadcast_ss (&(A[12] + 8)->real);
    left_r12_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 8)->imag));
    left_r12_c9_r = _mm256_broadcast_ss (&(A[12] + 9)->real);
    left_r12_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 9)->imag));
    left_r12_c10_r = _mm256_broadcast_ss (&(A[12] + 10)->real);
    left_r12_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 10)->imag));
    left_r12_c11_r = _mm256_broadcast_ss (&(A[12] + 11)->real);
    left_r12_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 11)->imag));
    left_r12_c12_r = _mm256_broadcast_ss (&(A[12] + 12)->real);
    left_r12_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 12)->imag));
    left_r12_c13_r = _mm256_broadcast_ss (&(A[12] + 13)->real);
    left_r12_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 13)->imag));
    left_r12_c14_r = _mm256_broadcast_ss (&(A[12] + 14)->real);
    left_r12_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 14)->imag));
    left_r12_c15_r = _mm256_broadcast_ss (&(A[12] + 15)->real);
    left_r12_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 15)->imag));
    left_r12_c16_r = _mm256_broadcast_ss (&(A[12] + 16)->real);
    left_r12_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 16)->imag));
    left_r12_c17_r = _mm256_broadcast_ss (&(A[12] + 17)->real);
    left_r12_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 17)->imag));
    left_r12_c18_r = _mm256_broadcast_ss (&(A[12] + 18)->real);
    left_r12_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[12] + 18)->imag));

    left_r13_c0_r = _mm256_broadcast_ss (&(A[13] + 0)->real);
    left_r13_c0_i = _mm256_broadcast_ss (&(A[13] + 0)->imag);
    left_r13_c1_r = _mm256_broadcast_ss (&(A[13] + 1)->real);
    left_r13_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 1)->imag));
    left_r13_c2_r = _mm256_broadcast_ss (&(A[13] + 2)->real);
    left_r13_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 2)->imag));
    left_r13_c3_r = _mm256_broadcast_ss (&(A[13] + 3)->real);
    left_r13_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 3)->imag));
    left_r13_c4_r = _mm256_broadcast_ss (&(A[13] + 4)->real);
    left_r13_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 4)->imag));
    left_r13_c5_r = _mm256_broadcast_ss (&(A[13] + 5)->real);
    left_r13_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 5)->imag));
    left_r13_c6_r = _mm256_broadcast_ss (&(A[13] + 6)->real);
    left_r13_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 6)->imag));
    left_r13_c7_r = _mm256_broadcast_ss (&(A[13] + 7)->real);
    left_r13_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 7)->imag));
    left_r13_c8_r = _mm256_broadcast_ss (&(A[13] + 8)->real);
    left_r13_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 8)->imag));
    left_r13_c9_r = _mm256_broadcast_ss (&(A[13] + 9)->real);
    left_r13_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 9)->imag));
    left_r13_c10_r = _mm256_broadcast_ss (&(A[13] + 10)->real);
    left_r13_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 10)->imag));
    left_r13_c11_r = _mm256_broadcast_ss (&(A[13] + 11)->real);
    left_r13_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 11)->imag));
    left_r13_c12_r = _mm256_broadcast_ss (&(A[13] + 12)->real);
    left_r13_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 12)->imag));
    left_r13_c13_r = _mm256_broadcast_ss (&(A[13] + 13)->real);
    left_r13_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 13)->imag));
    left_r13_c14_r = _mm256_broadcast_ss (&(A[13] + 14)->real);
    left_r13_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 14)->imag));
    left_r13_c15_r = _mm256_broadcast_ss (&(A[13] + 15)->real);
    left_r13_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 15)->imag));
    left_r13_c16_r = _mm256_broadcast_ss (&(A[13] + 16)->real);
    left_r13_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 16)->imag));
    left_r13_c17_r = _mm256_broadcast_ss (&(A[13] + 17)->real);
    left_r13_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 17)->imag));
    left_r13_c18_r = _mm256_broadcast_ss (&(A[13] + 18)->real);
    left_r13_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[13] + 18)->imag));

    left_r14_c0_r = _mm256_broadcast_ss (&(A[14] + 0)->real);
    left_r14_c0_i = _mm256_broadcast_ss (&(A[14] + 0)->imag);
    left_r14_c1_r = _mm256_broadcast_ss (&(A[14] + 1)->real);
    left_r14_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 1)->imag));
    left_r14_c2_r = _mm256_broadcast_ss (&(A[14] + 2)->real);
    left_r14_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 2)->imag));
    left_r14_c3_r = _mm256_broadcast_ss (&(A[14] + 3)->real);
    left_r14_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 3)->imag));
    left_r14_c4_r = _mm256_broadcast_ss (&(A[14] + 4)->real);
    left_r14_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 4)->imag));
    left_r14_c5_r = _mm256_broadcast_ss (&(A[14] + 5)->real);
    left_r14_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 5)->imag));
    left_r14_c6_r = _mm256_broadcast_ss (&(A[14] + 6)->real);
    left_r14_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 6)->imag));
    left_r14_c7_r = _mm256_broadcast_ss (&(A[14] + 7)->real);
    left_r14_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 7)->imag));
    left_r14_c8_r = _mm256_broadcast_ss (&(A[14] + 8)->real);
    left_r14_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 8)->imag));
    left_r14_c9_r = _mm256_broadcast_ss (&(A[14] + 9)->real);
    left_r14_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 9)->imag));
    left_r14_c10_r = _mm256_broadcast_ss (&(A[14] + 10)->real);
    left_r14_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 10)->imag));
    left_r14_c11_r = _mm256_broadcast_ss (&(A[14] + 11)->real);
    left_r14_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 11)->imag));
    left_r14_c12_r = _mm256_broadcast_ss (&(A[14] + 12)->real);
    left_r14_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 12)->imag));
    left_r14_c13_r = _mm256_broadcast_ss (&(A[14] + 13)->real);
    left_r14_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 13)->imag));
    left_r14_c14_r = _mm256_broadcast_ss (&(A[14] + 14)->real);
    left_r14_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 14)->imag));
    left_r14_c15_r = _mm256_broadcast_ss (&(A[14] + 15)->real);
    left_r14_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 15)->imag));
    left_r14_c16_r = _mm256_broadcast_ss (&(A[14] + 16)->real);
    left_r14_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 16)->imag));
    left_r14_c17_r = _mm256_broadcast_ss (&(A[14] + 17)->real);
    left_r14_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 17)->imag));
    left_r14_c18_r = _mm256_broadcast_ss (&(A[14] + 18)->real);
    left_r14_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[14] + 18)->imag));

    left_r15_c0_r = _mm256_broadcast_ss (&(A[15] + 0)->real);
    left_r15_c0_i = _mm256_broadcast_ss (&(A[15] + 0)->imag);
    left_r15_c1_r = _mm256_broadcast_ss (&(A[15] + 1)->real);
    left_r15_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 1)->imag));
    left_r15_c2_r = _mm256_broadcast_ss (&(A[15] + 2)->real);
    left_r15_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 2)->imag));
    left_r15_c3_r = _mm256_broadcast_ss (&(A[15] + 3)->real);
    left_r15_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 3)->imag));
    left_r15_c4_r = _mm256_broadcast_ss (&(A[15] + 4)->real);
    left_r15_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 4)->imag));
    left_r15_c5_r = _mm256_broadcast_ss (&(A[15] + 5)->real);
    left_r15_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 5)->imag));
    left_r15_c6_r = _mm256_broadcast_ss (&(A[15] + 6)->real);
    left_r15_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 6)->imag));
    left_r15_c7_r = _mm256_broadcast_ss (&(A[15] + 7)->real);
    left_r15_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 7)->imag));
    left_r15_c8_r = _mm256_broadcast_ss (&(A[15] + 8)->real);
    left_r15_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 8)->imag));
    left_r15_c9_r = _mm256_broadcast_ss (&(A[15] + 9)->real);
    left_r15_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 9)->imag));
    left_r15_c10_r = _mm256_broadcast_ss (&(A[15] + 10)->real);
    left_r15_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 10)->imag));
    left_r15_c11_r = _mm256_broadcast_ss (&(A[15] + 11)->real);
    left_r15_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 11)->imag));
    left_r15_c12_r = _mm256_broadcast_ss (&(A[15] + 12)->real);
    left_r15_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 12)->imag));
    left_r15_c13_r = _mm256_broadcast_ss (&(A[15] + 13)->real);
    left_r15_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 13)->imag));
    left_r15_c14_r = _mm256_broadcast_ss (&(A[15] + 14)->real);
    left_r15_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 14)->imag));
    left_r15_c15_r = _mm256_broadcast_ss (&(A[15] + 15)->real);
    left_r15_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 15)->imag));
    left_r15_c16_r = _mm256_broadcast_ss (&(A[15] + 16)->real);
    left_r15_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 16)->imag));
    left_r15_c17_r = _mm256_broadcast_ss (&(A[15] + 17)->real);
    left_r15_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 17)->imag));
    left_r15_c18_r = _mm256_broadcast_ss (&(A[15] + 18)->real);
    left_r15_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[15] + 18)->imag));

    left_r16_c0_r = _mm256_broadcast_ss (&(A[16] + 0)->real);
    left_r16_c0_i = _mm256_broadcast_ss (&(A[16] + 0)->imag);
    left_r16_c1_r = _mm256_broadcast_ss (&(A[16] + 1)->real);
    left_r16_c1_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 1)->imag));
    left_r16_c2_r = _mm256_broadcast_ss (&(A[16] + 2)->real);
    left_r16_c2_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 2)->imag));
    left_r16_c3_r = _mm256_broadcast_ss (&(A[16] + 3)->real);
    left_r16_c3_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 3)->imag));
    left_r16_c4_r = _mm256_broadcast_ss (&(A[16] + 4)->real);
    left_r16_c4_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 4)->imag));
    left_r16_c5_r = _mm256_broadcast_ss (&(A[16] + 5)->real);
    left_r16_c5_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 5)->imag));
    left_r16_c6_r = _mm256_broadcast_ss (&(A[16] + 6)->real);
    left_r16_c6_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 6)->imag));
    left_r16_c7_r = _mm256_broadcast_ss (&(A[16] + 7)->real);
    left_r16_c7_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 7)->imag));
    left_r16_c8_r = _mm256_broadcast_ss (&(A[16] + 8)->real);
    left_r16_c8_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 8)->imag));
    left_r16_c9_r = _mm256_broadcast_ss (&(A[16] + 9)->real);
    left_r16_c9_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 9)->imag));
    left_r16_c10_r = _mm256_broadcast_ss (&(A[16] + 10)->real);
    left_r16_c10_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 10)->imag));
    left_r16_c11_r = _mm256_broadcast_ss (&(A[16] + 11)->real);
    left_r16_c11_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 11)->imag));
    left_r16_c12_r = _mm256_broadcast_ss (&(A[16] + 12)->real);
    left_r16_c12_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 12)->imag));
    left_r16_c13_r = _mm256_broadcast_ss (&(A[16] + 13)->real);
    left_r16_c13_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 13)->imag));
    left_r16_c14_r = _mm256_broadcast_ss (&(A[16] + 14)->real);
    left_r16_c14_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 14)->imag));
    left_r16_c15_r = _mm256_broadcast_ss (&(A[16] + 15)->real);
    left_r16_c15_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 15)->imag));
    left_r16_c16_r = _mm256_broadcast_ss (&(A[16] + 16)->real);
    left_r16_c16_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 16)->imag));
    left_r16_c17_r = _mm256_broadcast_ss (&(A[16] + 17)->real);
    left_r16_c17_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 17)->imag));
    left_r16_c18_r = _mm256_broadcast_ss (&(A[16] + 18)->real);
    left_r16_c18_i = _mm256_mul_ps (negate_even_mult, _mm256_broadcast_ss (&(A[16] + 18)->imag));
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
        output_r7 = _mm256_fmaddsub_ps (               right_r_i, left_r7_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r7_c0_i));
        output_r8 = _mm256_fmaddsub_ps (               right_r_i, left_r8_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r8_c0_i));
        output_r9 = _mm256_fmaddsub_ps (               right_r_i, left_r9_c0_r,
                                        _mm256_mul_ps (right_i_r, left_r9_c0_i));
        output_r10 = _mm256_fmaddsub_ps (               right_r_i, left_r10_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r10_c0_i));
        output_r11 = _mm256_fmaddsub_ps (               right_r_i, left_r11_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r11_c0_i));
        output_r12 = _mm256_fmaddsub_ps (               right_r_i, left_r12_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r12_c0_i));
        output_r13 = _mm256_fmaddsub_ps (               right_r_i, left_r13_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r13_c0_i));
        output_r14 = _mm256_fmaddsub_ps (               right_r_i, left_r14_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r14_c0_i));
        output_r15 = _mm256_fmaddsub_ps (               right_r_i, left_r15_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r15_c0_i));
        output_r16 = _mm256_fmaddsub_ps (               right_r_i, left_r16_c0_r,
                                         _mm256_mul_ps (right_i_r, left_r16_c0_i));
        right_r_i = _mm256_loadu_ps (&(B[1] + c_c)->real);
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
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c1_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c1_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c1_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c1_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c1_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c1_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c1_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c1_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c1_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c1_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c1_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c1_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[2] + c_c)->real);
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
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c2_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c2_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c2_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c2_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c2_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c2_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c2_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c2_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c2_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c2_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c2_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c2_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[3] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c3_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c3_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c3_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c3_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c3_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c3_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c3_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c3_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c3_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c3_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c3_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c3_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c3_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c3_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c3_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c3_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c3_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c3_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c3_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[4] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c4_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c4_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c4_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c4_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c4_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c4_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c4_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c4_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c4_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c4_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c4_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c4_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c4_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c4_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c4_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c4_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c4_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c4_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c4_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[5] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c5_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c5_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c5_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c5_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c5_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c5_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c5_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c5_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c5_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c5_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c5_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c5_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c5_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c5_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c5_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c5_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c5_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c5_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c5_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[6] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c6_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c6_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c6_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c6_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c6_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c6_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c6_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c6_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c6_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c6_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c6_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c6_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c6_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c6_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c6_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c6_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c6_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c6_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c6_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[7] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c7_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c7_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c7_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c7_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c7_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c7_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c7_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c7_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c7_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c7_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c7_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c7_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c7_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c7_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c7_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c7_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c7_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c7_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c7_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[8] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c8_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c8_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c8_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c8_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c8_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c8_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c8_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c8_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c8_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c8_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c8_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c8_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c8_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c8_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c8_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c8_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c8_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c8_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c8_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[9] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c9_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c9_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c9_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c9_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c9_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c9_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c9_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c9_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c9_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c9_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c9_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c9_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c9_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c9_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c9_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c9_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c9_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c9_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c9_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[10] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c10_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c10_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c10_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c10_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c10_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c10_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c10_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c10_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c10_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c10_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c10_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c10_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c10_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c10_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c10_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c10_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c10_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c10_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c10_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[11] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c11_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c11_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c11_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c11_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c11_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c11_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c11_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c11_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c11_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c11_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c11_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c11_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c11_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c11_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c11_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c11_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c11_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c11_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c11_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[12] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c12_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c12_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c12_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c12_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c12_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c12_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c12_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c12_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c12_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c12_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c12_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c12_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c12_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c12_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c12_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c12_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c12_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c12_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c12_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[13] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c13_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c13_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c13_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c13_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c13_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c13_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c13_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c13_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c13_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c13_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c13_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c13_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c13_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c13_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c13_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c13_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c13_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c13_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c13_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[14] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c14_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c14_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c14_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c14_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c14_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c14_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c14_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c14_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c14_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c14_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c14_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c14_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c14_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c14_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c14_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c14_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c14_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c14_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c14_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[15] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c15_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c15_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c15_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c15_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c15_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c15_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c15_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c15_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c15_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c15_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c15_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c15_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c15_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c15_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c15_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c15_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c15_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c15_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c15_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[16] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c16_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c16_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c16_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c16_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c16_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c16_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c16_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c16_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c16_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c16_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c16_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c16_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c16_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c16_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c16_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c16_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c16_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c16_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c16_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[17] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_fmadd_ps (                 right_r_i, left_r0_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r0_c17_i, output_r0));
        output_r1 = _mm256_fmadd_ps (                 right_r_i, left_r1_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r1_c17_i, output_r1));
        output_r2 = _mm256_fmadd_ps (                 right_r_i, left_r2_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r2_c17_i, output_r2));
        output_r3 = _mm256_fmadd_ps (                 right_r_i, left_r3_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r3_c17_i, output_r3));
        output_r4 = _mm256_fmadd_ps (                 right_r_i, left_r4_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r4_c17_i, output_r4));
        output_r5 = _mm256_fmadd_ps (                 right_r_i, left_r5_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r5_c17_i, output_r5));
        output_r6 = _mm256_fmadd_ps (                 right_r_i, left_r6_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r6_c17_i, output_r6));
        output_r7 = _mm256_fmadd_ps (                 right_r_i, left_r7_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r7_c17_i, output_r7));
        output_r8 = _mm256_fmadd_ps (                 right_r_i, left_r8_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r8_c17_i, output_r8));
        output_r9 = _mm256_fmadd_ps (                 right_r_i, left_r9_c17_r,
                                     _mm256_fmadd_ps (right_i_r, left_r9_c17_i, output_r9));
        output_r10 = _mm256_fmadd_ps (                 right_r_i, left_r10_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r10_c17_i, output_r10));
        output_r11 = _mm256_fmadd_ps (                 right_r_i, left_r11_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r11_c17_i, output_r11));
        output_r12 = _mm256_fmadd_ps (                 right_r_i, left_r12_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r12_c17_i, output_r12));
        output_r13 = _mm256_fmadd_ps (                 right_r_i, left_r13_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r13_c17_i, output_r13));
        output_r14 = _mm256_fmadd_ps (                 right_r_i, left_r14_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r14_c17_i, output_r14));
        output_r15 = _mm256_fmadd_ps (                 right_r_i, left_r15_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r15_c17_i, output_r15));
        output_r16 = _mm256_fmadd_ps (                 right_r_i, left_r16_c17_r,
                                      _mm256_fmadd_ps (right_i_r, left_r16_c17_i, output_r16));
        right_r_i = _mm256_loadu_ps (&(B[18] + c_c)->real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_storeu_ps (&(C[0] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r0_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r0_c18_i, output_r0)));
        _mm256_storeu_ps (&(C[1] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r1_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r1_c18_i, output_r1)));
        _mm256_storeu_ps (&(C[2] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r2_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r2_c18_i, output_r2)));
        _mm256_storeu_ps (&(C[3] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r3_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r3_c18_i, output_r3)));
        _mm256_storeu_ps (&(C[4] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r4_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r4_c18_i, output_r4)));
        _mm256_storeu_ps (&(C[5] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r5_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r5_c18_i, output_r5)));
        _mm256_storeu_ps (&(C[6] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r6_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r6_c18_i, output_r6)));
        _mm256_storeu_ps (&(C[7] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r7_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r7_c18_i, output_r7)));
        _mm256_storeu_ps (&(C[8] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r8_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r8_c18_i, output_r8)));
        _mm256_storeu_ps (&(C[9] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r9_c18_r,
                                                                _mm256_fmadd_ps (right_i_r, left_r9_c18_i, output_r9)));
        _mm256_storeu_ps (&(C[10] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r10_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r10_c18_i, output_r10)));
        _mm256_storeu_ps (&(C[11] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r11_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r11_c18_i, output_r11)));
        _mm256_storeu_ps (&(C[12] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r12_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r12_c18_i, output_r12)));
        _mm256_storeu_ps (&(C[13] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r13_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r13_c18_i, output_r13)));
        _mm256_storeu_ps (&(C[14] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r14_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r14_c18_i, output_r14)));
        _mm256_storeu_ps (&(C[15] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r15_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r15_c18_i, output_r15)));
        _mm256_storeu_ps (&(C[16] + c_c)->real, _mm256_fmadd_ps (                 right_r_i, left_r16_c18_r,
                                                                 _mm256_fmadd_ps (right_i_r, left_r16_c18_i, output_r16)));
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

