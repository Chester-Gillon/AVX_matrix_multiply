/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

void cmat_mulx_avx_stripmine_nr_c_11_dot_product_length_14 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                            SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                            SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    const SAL_i32 nr_c = 11;
    /*const SAL_i32 dot_product_length = 14;*/
    const SAL_i32 L1_cache_size = 32768;
    const SAL_i32 strip_size_samples = (L1_cache_size / sizeof(__m256) / (nr_c + 2)) / 4 * 4;
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r, left_r3_c5_r, left_r3_c6_r, left_r3_c7_r, left_r3_c8_r, left_r3_c9_r, left_r3_c10_r, left_r3_c11_r, left_r3_c12_r, left_r3_c13_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i, left_r3_c5_i, left_r3_c6_i, left_r3_c7_i, left_r3_c8_i, left_r3_c9_i, left_r3_c10_i, left_r3_c11_i, left_r3_c12_i, left_r3_c13_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r, left_r4_c4_r, left_r4_c5_r, left_r4_c6_r, left_r4_c7_r, left_r4_c8_r, left_r4_c9_r, left_r4_c10_r, left_r4_c11_r, left_r4_c12_r, left_r4_c13_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i, left_r4_c4_i, left_r4_c5_i, left_r4_c6_i, left_r4_c7_i, left_r4_c8_i, left_r4_c9_i, left_r4_c10_i, left_r4_c11_i, left_r4_c12_i, left_r4_c13_i;
    __m256 left_r5_c0_r, left_r5_c1_r, left_r5_c2_r, left_r5_c3_r, left_r5_c4_r, left_r5_c5_r, left_r5_c6_r, left_r5_c7_r, left_r5_c8_r, left_r5_c9_r, left_r5_c10_r, left_r5_c11_r, left_r5_c12_r, left_r5_c13_r;
    __m256 left_r5_c0_i, left_r5_c1_i, left_r5_c2_i, left_r5_c3_i, left_r5_c4_i, left_r5_c5_i, left_r5_c6_i, left_r5_c7_i, left_r5_c8_i, left_r5_c9_i, left_r5_c10_i, left_r5_c11_i, left_r5_c12_i, left_r5_c13_i;
    __m256 left_r6_c0_r, left_r6_c1_r, left_r6_c2_r, left_r6_c3_r, left_r6_c4_r, left_r6_c5_r, left_r6_c6_r, left_r6_c7_r, left_r6_c8_r, left_r6_c9_r, left_r6_c10_r, left_r6_c11_r, left_r6_c12_r, left_r6_c13_r;
    __m256 left_r6_c0_i, left_r6_c1_i, left_r6_c2_i, left_r6_c3_i, left_r6_c4_i, left_r6_c5_i, left_r6_c6_i, left_r6_c7_i, left_r6_c8_i, left_r6_c9_i, left_r6_c10_i, left_r6_c11_i, left_r6_c12_i, left_r6_c13_i;
    __m256 left_r7_c0_r, left_r7_c1_r, left_r7_c2_r, left_r7_c3_r, left_r7_c4_r, left_r7_c5_r, left_r7_c6_r, left_r7_c7_r, left_r7_c8_r, left_r7_c9_r, left_r7_c10_r, left_r7_c11_r, left_r7_c12_r, left_r7_c13_r;
    __m256 left_r7_c0_i, left_r7_c1_i, left_r7_c2_i, left_r7_c3_i, left_r7_c4_i, left_r7_c5_i, left_r7_c6_i, left_r7_c7_i, left_r7_c8_i, left_r7_c9_i, left_r7_c10_i, left_r7_c11_i, left_r7_c12_i, left_r7_c13_i;
    __m256 left_r8_c0_r, left_r8_c1_r, left_r8_c2_r, left_r8_c3_r, left_r8_c4_r, left_r8_c5_r, left_r8_c6_r, left_r8_c7_r, left_r8_c8_r, left_r8_c9_r, left_r8_c10_r, left_r8_c11_r, left_r8_c12_r, left_r8_c13_r;
    __m256 left_r8_c0_i, left_r8_c1_i, left_r8_c2_i, left_r8_c3_i, left_r8_c4_i, left_r8_c5_i, left_r8_c6_i, left_r8_c7_i, left_r8_c8_i, left_r8_c9_i, left_r8_c10_i, left_r8_c11_i, left_r8_c12_i, left_r8_c13_i;
    __m256 left_r9_c0_r, left_r9_c1_r, left_r9_c2_r, left_r9_c3_r, left_r9_c4_r, left_r9_c5_r, left_r9_c6_r, left_r9_c7_r, left_r9_c8_r, left_r9_c9_r, left_r9_c10_r, left_r9_c11_r, left_r9_c12_r, left_r9_c13_r;
    __m256 left_r9_c0_i, left_r9_c1_i, left_r9_c2_i, left_r9_c3_i, left_r9_c4_i, left_r9_c5_i, left_r9_c6_i, left_r9_c7_i, left_r9_c8_i, left_r9_c9_i, left_r9_c10_i, left_r9_c11_i, left_r9_c12_i, left_r9_c13_i;
    __m256 left_r10_c0_r, left_r10_c1_r, left_r10_c2_r, left_r10_c3_r, left_r10_c4_r, left_r10_c5_r, left_r10_c6_r, left_r10_c7_r, left_r10_c8_r, left_r10_c9_r, left_r10_c10_r, left_r10_c11_r, left_r10_c12_r, left_r10_c13_r;
    __m256 left_r10_c0_i, left_r10_c1_i, left_r10_c2_i, left_r10_c3_i, left_r10_c4_i, left_r10_c5_i, left_r10_c6_i, left_r10_c7_i, left_r10_c8_i, left_r10_c9_i, left_r10_c10_i, left_r10_c11_i, left_r10_c12_i, left_r10_c13_i;
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
    left_r4_c7_r = _mm256_broadcast_ss (&(A[4] + 7)->real);
    left_r4_c7_i = _mm256_broadcast_ss (&(A[4] + 7)->imag);
    left_r4_c8_r = _mm256_broadcast_ss (&(A[4] + 8)->real);
    left_r4_c8_i = _mm256_broadcast_ss (&(A[4] + 8)->imag);
    left_r4_c9_r = _mm256_broadcast_ss (&(A[4] + 9)->real);
    left_r4_c9_i = _mm256_broadcast_ss (&(A[4] + 9)->imag);
    left_r4_c10_r = _mm256_broadcast_ss (&(A[4] + 10)->real);
    left_r4_c10_i = _mm256_broadcast_ss (&(A[4] + 10)->imag);
    left_r4_c11_r = _mm256_broadcast_ss (&(A[4] + 11)->real);
    left_r4_c11_i = _mm256_broadcast_ss (&(A[4] + 11)->imag);
    left_r4_c12_r = _mm256_broadcast_ss (&(A[4] + 12)->real);
    left_r4_c12_i = _mm256_broadcast_ss (&(A[4] + 12)->imag);
    left_r4_c13_r = _mm256_broadcast_ss (&(A[4] + 13)->real);
    left_r4_c13_i = _mm256_broadcast_ss (&(A[4] + 13)->imag);

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
    left_r5_c7_r = _mm256_broadcast_ss (&(A[5] + 7)->real);
    left_r5_c7_i = _mm256_broadcast_ss (&(A[5] + 7)->imag);
    left_r5_c8_r = _mm256_broadcast_ss (&(A[5] + 8)->real);
    left_r5_c8_i = _mm256_broadcast_ss (&(A[5] + 8)->imag);
    left_r5_c9_r = _mm256_broadcast_ss (&(A[5] + 9)->real);
    left_r5_c9_i = _mm256_broadcast_ss (&(A[5] + 9)->imag);
    left_r5_c10_r = _mm256_broadcast_ss (&(A[5] + 10)->real);
    left_r5_c10_i = _mm256_broadcast_ss (&(A[5] + 10)->imag);
    left_r5_c11_r = _mm256_broadcast_ss (&(A[5] + 11)->real);
    left_r5_c11_i = _mm256_broadcast_ss (&(A[5] + 11)->imag);
    left_r5_c12_r = _mm256_broadcast_ss (&(A[5] + 12)->real);
    left_r5_c12_i = _mm256_broadcast_ss (&(A[5] + 12)->imag);
    left_r5_c13_r = _mm256_broadcast_ss (&(A[5] + 13)->real);
    left_r5_c13_i = _mm256_broadcast_ss (&(A[5] + 13)->imag);

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
    left_r6_c5_r = _mm256_broadcast_ss (&(A[6] + 5)->real);
    left_r6_c5_i = _mm256_broadcast_ss (&(A[6] + 5)->imag);
    left_r6_c6_r = _mm256_broadcast_ss (&(A[6] + 6)->real);
    left_r6_c6_i = _mm256_broadcast_ss (&(A[6] + 6)->imag);
    left_r6_c7_r = _mm256_broadcast_ss (&(A[6] + 7)->real);
    left_r6_c7_i = _mm256_broadcast_ss (&(A[6] + 7)->imag);
    left_r6_c8_r = _mm256_broadcast_ss (&(A[6] + 8)->real);
    left_r6_c8_i = _mm256_broadcast_ss (&(A[6] + 8)->imag);
    left_r6_c9_r = _mm256_broadcast_ss (&(A[6] + 9)->real);
    left_r6_c9_i = _mm256_broadcast_ss (&(A[6] + 9)->imag);
    left_r6_c10_r = _mm256_broadcast_ss (&(A[6] + 10)->real);
    left_r6_c10_i = _mm256_broadcast_ss (&(A[6] + 10)->imag);
    left_r6_c11_r = _mm256_broadcast_ss (&(A[6] + 11)->real);
    left_r6_c11_i = _mm256_broadcast_ss (&(A[6] + 11)->imag);
    left_r6_c12_r = _mm256_broadcast_ss (&(A[6] + 12)->real);
    left_r6_c12_i = _mm256_broadcast_ss (&(A[6] + 12)->imag);
    left_r6_c13_r = _mm256_broadcast_ss (&(A[6] + 13)->real);
    left_r6_c13_i = _mm256_broadcast_ss (&(A[6] + 13)->imag);

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
    left_r7_c5_r = _mm256_broadcast_ss (&(A[7] + 5)->real);
    left_r7_c5_i = _mm256_broadcast_ss (&(A[7] + 5)->imag);
    left_r7_c6_r = _mm256_broadcast_ss (&(A[7] + 6)->real);
    left_r7_c6_i = _mm256_broadcast_ss (&(A[7] + 6)->imag);
    left_r7_c7_r = _mm256_broadcast_ss (&(A[7] + 7)->real);
    left_r7_c7_i = _mm256_broadcast_ss (&(A[7] + 7)->imag);
    left_r7_c8_r = _mm256_broadcast_ss (&(A[7] + 8)->real);
    left_r7_c8_i = _mm256_broadcast_ss (&(A[7] + 8)->imag);
    left_r7_c9_r = _mm256_broadcast_ss (&(A[7] + 9)->real);
    left_r7_c9_i = _mm256_broadcast_ss (&(A[7] + 9)->imag);
    left_r7_c10_r = _mm256_broadcast_ss (&(A[7] + 10)->real);
    left_r7_c10_i = _mm256_broadcast_ss (&(A[7] + 10)->imag);
    left_r7_c11_r = _mm256_broadcast_ss (&(A[7] + 11)->real);
    left_r7_c11_i = _mm256_broadcast_ss (&(A[7] + 11)->imag);
    left_r7_c12_r = _mm256_broadcast_ss (&(A[7] + 12)->real);
    left_r7_c12_i = _mm256_broadcast_ss (&(A[7] + 12)->imag);
    left_r7_c13_r = _mm256_broadcast_ss (&(A[7] + 13)->real);
    left_r7_c13_i = _mm256_broadcast_ss (&(A[7] + 13)->imag);

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
    left_r8_c5_r = _mm256_broadcast_ss (&(A[8] + 5)->real);
    left_r8_c5_i = _mm256_broadcast_ss (&(A[8] + 5)->imag);
    left_r8_c6_r = _mm256_broadcast_ss (&(A[8] + 6)->real);
    left_r8_c6_i = _mm256_broadcast_ss (&(A[8] + 6)->imag);
    left_r8_c7_r = _mm256_broadcast_ss (&(A[8] + 7)->real);
    left_r8_c7_i = _mm256_broadcast_ss (&(A[8] + 7)->imag);
    left_r8_c8_r = _mm256_broadcast_ss (&(A[8] + 8)->real);
    left_r8_c8_i = _mm256_broadcast_ss (&(A[8] + 8)->imag);
    left_r8_c9_r = _mm256_broadcast_ss (&(A[8] + 9)->real);
    left_r8_c9_i = _mm256_broadcast_ss (&(A[8] + 9)->imag);
    left_r8_c10_r = _mm256_broadcast_ss (&(A[8] + 10)->real);
    left_r8_c10_i = _mm256_broadcast_ss (&(A[8] + 10)->imag);
    left_r8_c11_r = _mm256_broadcast_ss (&(A[8] + 11)->real);
    left_r8_c11_i = _mm256_broadcast_ss (&(A[8] + 11)->imag);
    left_r8_c12_r = _mm256_broadcast_ss (&(A[8] + 12)->real);
    left_r8_c12_i = _mm256_broadcast_ss (&(A[8] + 12)->imag);
    left_r8_c13_r = _mm256_broadcast_ss (&(A[8] + 13)->real);
    left_r8_c13_i = _mm256_broadcast_ss (&(A[8] + 13)->imag);

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
    left_r9_c5_r = _mm256_broadcast_ss (&(A[9] + 5)->real);
    left_r9_c5_i = _mm256_broadcast_ss (&(A[9] + 5)->imag);
    left_r9_c6_r = _mm256_broadcast_ss (&(A[9] + 6)->real);
    left_r9_c6_i = _mm256_broadcast_ss (&(A[9] + 6)->imag);
    left_r9_c7_r = _mm256_broadcast_ss (&(A[9] + 7)->real);
    left_r9_c7_i = _mm256_broadcast_ss (&(A[9] + 7)->imag);
    left_r9_c8_r = _mm256_broadcast_ss (&(A[9] + 8)->real);
    left_r9_c8_i = _mm256_broadcast_ss (&(A[9] + 8)->imag);
    left_r9_c9_r = _mm256_broadcast_ss (&(A[9] + 9)->real);
    left_r9_c9_i = _mm256_broadcast_ss (&(A[9] + 9)->imag);
    left_r9_c10_r = _mm256_broadcast_ss (&(A[9] + 10)->real);
    left_r9_c10_i = _mm256_broadcast_ss (&(A[9] + 10)->imag);
    left_r9_c11_r = _mm256_broadcast_ss (&(A[9] + 11)->real);
    left_r9_c11_i = _mm256_broadcast_ss (&(A[9] + 11)->imag);
    left_r9_c12_r = _mm256_broadcast_ss (&(A[9] + 12)->real);
    left_r9_c12_i = _mm256_broadcast_ss (&(A[9] + 12)->imag);
    left_r9_c13_r = _mm256_broadcast_ss (&(A[9] + 13)->real);
    left_r9_c13_i = _mm256_broadcast_ss (&(A[9] + 13)->imag);

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
    left_r10_c5_r = _mm256_broadcast_ss (&(A[10] + 5)->real);
    left_r10_c5_i = _mm256_broadcast_ss (&(A[10] + 5)->imag);
    left_r10_c6_r = _mm256_broadcast_ss (&(A[10] + 6)->real);
    left_r10_c6_i = _mm256_broadcast_ss (&(A[10] + 6)->imag);
    left_r10_c7_r = _mm256_broadcast_ss (&(A[10] + 7)->real);
    left_r10_c7_i = _mm256_broadcast_ss (&(A[10] + 7)->imag);
    left_r10_c8_r = _mm256_broadcast_ss (&(A[10] + 8)->real);
    left_r10_c8_i = _mm256_broadcast_ss (&(A[10] + 8)->imag);
    left_r10_c9_r = _mm256_broadcast_ss (&(A[10] + 9)->real);
    left_r10_c9_i = _mm256_broadcast_ss (&(A[10] + 9)->imag);
    left_r10_c10_r = _mm256_broadcast_ss (&(A[10] + 10)->real);
    left_r10_c10_i = _mm256_broadcast_ss (&(A[10] + 10)->imag);
    left_r10_c11_r = _mm256_broadcast_ss (&(A[10] + 11)->real);
    left_r10_c11_i = _mm256_broadcast_ss (&(A[10] + 11)->imag);
    left_r10_c12_r = _mm256_broadcast_ss (&(A[10] + 12)->real);
    left_r10_c12_i = _mm256_broadcast_ss (&(A[10] + 12)->imag);
    left_r10_c13_r = _mm256_broadcast_ss (&(A[10] + 13)->real);
    left_r10_c13_i = _mm256_broadcast_ss (&(A[10] + 13)->imag);
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
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c2_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c2_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c2_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c2_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c2_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c2_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c2_i))));
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
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c3_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c3_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c3_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c3_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c3_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c3_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c3_i))));
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
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c4_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c4_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c4_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c4_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c4_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c4_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c4_i))));
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
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c5_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c5_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c5_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c5_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c5_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c5_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c5_i))));
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
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c6_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c6_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c6_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c6_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c6_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c6_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c6_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[7] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c7_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c7_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c7_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c7_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c7_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c7_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c7_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c7_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c7_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c7_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c7_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c7_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c7_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[8] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c8_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c8_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c8_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c8_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c8_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c8_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c8_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c8_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c8_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c8_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c8_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c8_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c8_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[9] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c9_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c9_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c9_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c9_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c9_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c9_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c9_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c9_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c9_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c9_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c9_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c9_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c9_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[10] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c10_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c10_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c10_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c10_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c10_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c10_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c10_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c10_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c10_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c10_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c10_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c10_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c10_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[11] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c11_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c11_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c11_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c11_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c11_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c11_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c11_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c11_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c11_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c11_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c11_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c11_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c11_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[12] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c12_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c12_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c12_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c12_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c12_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c12_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c12_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c12_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c12_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c12_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c12_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c12_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c12_i))));
        }
        for (c_c = strip_start_col; c_c < next_strip_col; c_c += 4)
        {
            right_r_i = _mm256_load_ps (&(B[13] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            _mm256_store_ps (&(C[0] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[0] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r0_c13_i))));
            _mm256_store_ps (&(C[1] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[1] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r1_c13_i))));
            _mm256_store_ps (&(C[2] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[2] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r2_c13_i))));
            _mm256_store_ps (&(C[3] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[3] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r3_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r3_c13_i))));
            _mm256_store_ps (&(C[4] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[4] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r4_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r4_c13_i))));
            _mm256_store_ps (&(C[5] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[5] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r5_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r5_c13_i))));
            _mm256_store_ps (&(C[6] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[6] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r6_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r6_c13_i))));
            _mm256_store_ps (&(C[7] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[7] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r7_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r7_c13_i))));
            _mm256_store_ps (&(C[8] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[8] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r8_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r8_c13_i))));
            _mm256_store_ps (&(C[9] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[9] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r9_c13_r),
                                                                                                                        _mm256_mul_ps (right_i_r, left_r9_c13_i))));
            _mm256_store_ps (&(C[10] + c_c)->real, _mm256_add_ps (_mm256_load_ps(&(C[10] + c_c)->real), _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r10_c13_r),
                                                                                                                          _mm256_mul_ps (right_i_r, left_r10_c13_i))));
        }
        strip_start_col = next_strip_col;
    }
#ifdef IACA_OPERATE
    IACA_END
#endif
}

