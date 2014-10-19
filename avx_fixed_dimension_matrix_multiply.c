#include <immintrin.h>
#include <sal.h>

#include "avx_fixed_dimension_matrix_multiply.h"

#define SWAP_REAL_IMAG_PERMUTE 0xB1

SAL_i32 cmat_mulx_avx_nr_c_5_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                               SAL_i32 A_tcols, 	/* left input column stride */
  	                                               SAL_cf32 *B, 	/* right input matrix */
  	                                               SAL_i32 B_tcols, 	/* right input column stride */
  	                                               SAL_cf32 *C, 	/* output matrix */
  	                                               SAL_i32 C_tcols, 	/* output column stride */
  	                                               SAL_i32 nc_c) 	/* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i;
    __m256 left_r3_c0_r, left_r3_c1_r, left_r3_c2_r, left_r3_c3_r, left_r3_c4_r, left_r3_c5_r, left_r3_c6_r, left_r3_c7_r;
    __m256 left_r3_c0_i, left_r3_c1_i, left_r3_c2_i, left_r3_c3_i, left_r3_c4_i, left_r3_c5_i, left_r3_c6_i, left_r3_c7_i;
    __m256 left_r4_c0_r, left_r4_c1_r, left_r4_c2_r, left_r4_c3_r, left_r4_c4_r, left_r4_c5_r, left_r4_c6_r, left_r4_c7_r;
    __m256 left_r4_c0_i, left_r4_c1_i, left_r4_c2_i, left_r4_c3_i, left_r4_c4_i, left_r4_c5_i, left_r4_c6_i, left_r4_c7_i;
    __m256 right_r0_r_i, right_r0_i_r;
    __m256 right_r1_r_i, right_r1_i_r;
    __m256 right_r2_r_i, right_r2_i_r;
    __m256 right_r3_r_i, right_r3_i_r;
    __m256 right_r4_r_i, right_r4_i_r;
    __m256 right_r5_r_i, right_r5_i_r;
    __m256 right_r6_r_i, right_r6_i_r;
    __m256 right_r7_r_i, right_r7_i_r;
    SAL_i32 c_c;

    left_r0_c0_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].real);
    left_r0_c0_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].imag);
    left_r0_c1_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].real);
    left_r0_c1_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].imag);
    left_r0_c2_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].real);
    left_r0_c2_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].imag);
    left_r0_c3_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 3].real);
    left_r0_c3_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 3].imag);
    left_r0_c4_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 4].real);
    left_r0_c4_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 4].imag);
    left_r0_c5_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 5].real);
    left_r0_c5_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 5].imag);
    left_r0_c6_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 6].real);
    left_r0_c6_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 6].imag);
    left_r0_c7_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 7].real);
    left_r0_c7_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 7].imag);

    left_r1_c0_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].real);
    left_r1_c0_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].imag);
    left_r1_c1_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].real);
    left_r1_c1_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].imag);
    left_r1_c2_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].real);
    left_r1_c2_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].imag);
    left_r1_c3_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 3].real);
    left_r1_c3_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 3].imag);
    left_r1_c4_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 4].real);
    left_r1_c4_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 4].imag);
    left_r1_c5_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 5].real);
    left_r1_c5_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 5].imag);
    left_r1_c6_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 6].real);
    left_r1_c6_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 6].imag);
    left_r1_c7_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 7].real);
    left_r1_c7_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 7].imag);

    left_r2_c0_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].real);
    left_r2_c0_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].imag);
    left_r2_c1_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].real);
    left_r2_c1_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].imag);
    left_r2_c2_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].real);
    left_r2_c2_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].imag);
    left_r2_c3_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 3].real);
    left_r2_c3_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 3].imag);
    left_r2_c4_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 4].real);
    left_r2_c4_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 4].imag);
    left_r2_c5_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 5].real);
    left_r2_c5_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 5].imag);
    left_r2_c6_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 6].real);
    left_r2_c6_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 6].imag);
    left_r2_c7_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 7].real);
    left_r2_c7_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 7].imag);

    left_r3_c0_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 0].real);
    left_r3_c0_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 0].imag);
    left_r3_c1_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 1].real);
    left_r3_c1_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 1].imag);
    left_r3_c2_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 2].real);
    left_r3_c2_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 2].imag);
    left_r3_c3_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 3].real);
    left_r3_c3_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 3].imag);
    left_r3_c4_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 4].real);
    left_r3_c4_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 4].imag);
    left_r3_c5_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 5].real);
    left_r3_c5_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 5].imag);
    left_r3_c6_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 6].real);
    left_r3_c6_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 6].imag);
    left_r3_c7_r = _mm256_broadcast_ss (&A[(3 * A_tcols) + 7].real);
    left_r3_c7_i = _mm256_broadcast_ss (&A[(3 * A_tcols) + 7].imag);

    left_r4_c0_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 0].real);
    left_r4_c0_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 0].imag);
    left_r4_c1_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 1].real);
    left_r4_c1_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 1].imag);
    left_r4_c2_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 2].real);
    left_r4_c2_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 2].imag);
    left_r4_c3_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 3].real);
    left_r4_c3_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 3].imag);
    left_r4_c4_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 4].real);
    left_r4_c4_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 4].imag);
    left_r4_c5_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 5].real);
    left_r4_c5_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 5].imag);
    left_r4_c6_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 6].real);
    left_r4_c6_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 6].imag);
    left_r4_c7_r = _mm256_broadcast_ss (&A[(4 * A_tcols) + 7].real);
    left_r4_c7_i = _mm256_broadcast_ss (&A[(4 * A_tcols) + 7].imag);
    
    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r0_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_r0_i_r = _mm256_permute_ps (right_r0_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r1_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_r1_i_r = _mm256_permute_ps (right_r1_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r2_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_r2_i_r = _mm256_permute_ps (right_r2_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r3_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_r3_i_r = _mm256_permute_ps (right_r3_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r4_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_r4_i_r = _mm256_permute_ps (right_r4_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r5_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_r5_i_r = _mm256_permute_ps (right_r5_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r6_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_r6_i_r = _mm256_permute_ps (right_r6_r_i, SWAP_REAL_IMAG_PERMUTE);
        right_r7_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_r7_i_r = _mm256_permute_ps (right_r7_r_i, SWAP_REAL_IMAG_PERMUTE);

        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r0_c0_r),
                                                      _mm256_mul_ps (right_r0_i_r, left_r0_c0_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r0_c1_r),
                                                      _mm256_mul_ps (right_r1_i_r, left_r0_c1_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r0_c2_r),
                                                      _mm256_mul_ps (right_r2_i_r, left_r0_c2_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r0_c3_r),
                                                      _mm256_mul_ps (right_r3_i_r, left_r0_c3_i)))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r0_c4_r),
                                                      _mm256_mul_ps (right_r4_i_r, left_r0_c4_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r5_r_i, left_r0_c5_r),
                                                      _mm256_mul_ps (right_r5_i_r, left_r0_c5_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r6_r_i, left_r0_c6_r),
                                                      _mm256_mul_ps (right_r6_i_r, left_r0_c6_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r7_r_i, left_r0_c7_r),
                                                      _mm256_mul_ps (right_r7_i_r, left_r0_c7_i))))));

        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r1_c0_r),
                                                      _mm256_mul_ps (right_r0_i_r, left_r1_c0_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r1_c1_r),
                                                      _mm256_mul_ps (right_r1_i_r, left_r1_c1_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r1_c2_r),
                                                      _mm256_mul_ps (right_r2_i_r, left_r1_c2_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r1_c3_r),
                                                      _mm256_mul_ps (right_r3_i_r, left_r1_c3_i)))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r1_c4_r),
                                                      _mm256_mul_ps (right_r4_i_r, left_r1_c4_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r5_r_i, left_r1_c5_r),
                                                      _mm256_mul_ps (right_r5_i_r, left_r1_c5_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r6_r_i, left_r1_c6_r),
                                                      _mm256_mul_ps (right_r6_i_r, left_r1_c6_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r7_r_i, left_r1_c7_r),
                                                      _mm256_mul_ps (right_r7_i_r, left_r1_c7_i))))));

        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r2_c0_r),
                                                      _mm256_mul_ps (right_r0_i_r, left_r2_c0_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r2_c1_r),
                                                      _mm256_mul_ps (right_r1_i_r, left_r2_c1_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r2_c2_r),
                                                      _mm256_mul_ps (right_r2_i_r, left_r2_c2_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r2_c3_r),
                                                      _mm256_mul_ps (right_r3_i_r, left_r2_c3_i)))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r2_c4_r),
                                                      _mm256_mul_ps (right_r4_i_r, left_r2_c4_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r5_r_i, left_r2_c5_r),
                                                      _mm256_mul_ps (right_r5_i_r, left_r2_c5_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r6_r_i, left_r2_c6_r),
                                                      _mm256_mul_ps (right_r6_i_r, left_r2_c6_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r7_r_i, left_r2_c7_r),
                                                      _mm256_mul_ps (right_r7_i_r, left_r2_c7_i))))));

        _mm256_store_ps (&C[(3 * C_tcols) + c_c].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r3_c0_r),
                                                      _mm256_mul_ps (right_r0_i_r, left_r3_c0_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r3_c1_r),
                                                      _mm256_mul_ps (right_r1_i_r, left_r3_c1_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r3_c2_r),
                                                      _mm256_mul_ps (right_r2_i_r, left_r3_c2_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r3_c3_r),
                                                      _mm256_mul_ps (right_r3_i_r, left_r3_c3_i)))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r3_c4_r),
                                                      _mm256_mul_ps (right_r4_i_r, left_r3_c4_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r5_r_i, left_r3_c5_r),
                                                      _mm256_mul_ps (right_r5_i_r, left_r3_c5_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r6_r_i, left_r3_c6_r),
                                                      _mm256_mul_ps (right_r6_i_r, left_r3_c6_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r7_r_i, left_r3_c7_r),
                                                      _mm256_mul_ps (right_r7_i_r, left_r3_c7_i))))));

        _mm256_store_ps (&C[(4 * C_tcols) + c_c].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r0_r_i, left_r4_c0_r),
                                                      _mm256_mul_ps (right_r0_i_r, left_r4_c0_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r1_r_i, left_r4_c1_r),
                                                      _mm256_mul_ps (right_r1_i_r, left_r4_c1_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r2_r_i, left_r4_c2_r),
                                                      _mm256_mul_ps (right_r2_i_r, left_r4_c2_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r3_r_i, left_r4_c3_r),
                                                      _mm256_mul_ps (right_r3_i_r, left_r4_c3_i)))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r4_r_i, left_r4_c4_r),
                                                      _mm256_mul_ps (right_r4_i_r, left_r4_c4_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r5_r_i, left_r4_c5_r),
                                                      _mm256_mul_ps (right_r5_i_r, left_r4_c5_i))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (right_r6_r_i, left_r4_c6_r),
                                                      _mm256_mul_ps (right_r6_i_r, left_r4_c6_i)),
                                    _mm256_addsub_ps (_mm256_mul_ps (right_r7_r_i, left_r4_c7_r),
                                                      _mm256_mul_ps (right_r7_i_r, left_r4_c7_i))))));
    }

    return SAL_SUCCESS;
}
