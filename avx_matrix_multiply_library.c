
/* @todo
 * Initially coded using the Intel vector types and intrinsics as described in:
 *   https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 *
 * Uses load/stores which require 32-byte alignment for samples and output arrays.
 * http://code.google.com/p/dynamorio/issues/detail?id=438 suggests that on aligned data
 * movaps and movapd take about the same time, but but movaps takes one less byte to encode the instruction.
 * Allowing unaligned data gives more freedom on input strides, but at the expense of a slowdown.
 *
 * http://www.thinkingandcomputing.com/2014/02/28/using-avx-instructions-in-matrix-multiplication/ is
 * an example optimisation which has variables named after the vector registers and re-uses the
 * variables in an attempt to maximise register usage?
 */

#include <immintrin.h>
#include <sal.h>

#include "avx_matrix_multiply_library.h"

#define SWAP_REAL_IMAG_PERMUTE 0xB1

SAL_i32 zmat_mulx_avx_dot_product_length_8 (SAL_zf32 *A, 	/* left input matrix */
  	                                        SAL_i32 A_tcols, 	/* left input column stride */
  	                                        SAL_zf32 *B, 	/* right input matrix */
  	                                        SAL_i32 B_tcols, 	/* right input column stride */
  	                                        SAL_zf32 *C, 	/* output matrix */
  	                                        SAL_i32 nr_c, 	/* rows count in C */
                                            SAL_i32 C_tcols, 	/* output column stride */
  	                                        SAL_i32 nc_c) 	/* column count in C */
{
    __m256 A_c0_real[NR_C_MAX], A_c0_imag[NR_C_MAX];
    __m256 A_c1_real[NR_C_MAX], A_c1_imag[NR_C_MAX];
    __m256 A_c2_real[NR_C_MAX], A_c2_imag[NR_C_MAX];
    __m256 A_c3_real[NR_C_MAX], A_c3_imag[NR_C_MAX];
    __m256 A_c4_real[NR_C_MAX], A_c4_imag[NR_C_MAX];
    __m256 A_c5_real[NR_C_MAX], A_c5_imag[NR_C_MAX];
    __m256 A_c6_real[NR_C_MAX], A_c6_imag[NR_C_MAX];
    __m256 A_c7_real[NR_C_MAX], A_c7_imag[NR_C_MAX];
    __m256 B_r0_real, B_r0_imag;
    __m256 B_r1_real, B_r1_imag;
    __m256 B_r2_real, B_r2_imag;
    __m256 B_r3_real, B_r3_imag;
    __m256 B_r4_real, B_r4_imag;
    __m256 B_r5_real, B_r5_imag;
    __m256 B_r6_real, B_r6_imag;
    __m256 B_r7_real, B_r7_imag;
    SAL_i32 r_c_index, c_c_index;
    
    for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
    {
        A_c0_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 0]);
        A_c0_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 0]);
        A_c1_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 1]);
        A_c1_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 1]);
        A_c2_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 2]);
        A_c2_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 2]);
        A_c3_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 3]);
        A_c3_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 3]);
        A_c4_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 4]);
        A_c4_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 4]);
        A_c5_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 5]);
        A_c5_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 5]);
        A_c6_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 6]);
        A_c6_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 6]);
        A_c7_real[r_c_index] = _mm256_broadcast_ss (&A->realp[(r_c_index * A_tcols) + 7]);
        A_c7_imag[r_c_index] = _mm256_broadcast_ss (&A->imagp[(r_c_index * A_tcols) + 7]);
    }
    
    for (c_c_index = 0; c_c_index < nc_c; c_c_index += 8)
    {
        B_r0_real = _mm256_loadu_ps (&B->realp[(0 * B_tcols) + c_c_index]);
        B_r0_imag = _mm256_loadu_ps (&B->imagp[(0 * B_tcols) + c_c_index]);
        B_r1_real = _mm256_loadu_ps (&B->realp[(1 * B_tcols) + c_c_index]);
        B_r1_imag = _mm256_loadu_ps (&B->imagp[(1 * B_tcols) + c_c_index]);
        B_r2_real = _mm256_loadu_ps (&B->realp[(2 * B_tcols) + c_c_index]);
        B_r2_imag = _mm256_loadu_ps (&B->imagp[(2 * B_tcols) + c_c_index]);
        B_r3_real = _mm256_loadu_ps (&B->realp[(3 * B_tcols) + c_c_index]);
        B_r3_imag = _mm256_loadu_ps (&B->imagp[(3 * B_tcols) + c_c_index]);
        B_r4_real = _mm256_loadu_ps (&B->realp[(4 * B_tcols) + c_c_index]);
        B_r4_imag = _mm256_loadu_ps (&B->imagp[(4 * B_tcols) + c_c_index]);
        B_r5_real = _mm256_loadu_ps (&B->realp[(5 * B_tcols) + c_c_index]);
        B_r5_imag = _mm256_loadu_ps (&B->imagp[(5 * B_tcols) + c_c_index]);
        B_r6_real = _mm256_loadu_ps (&B->realp[(6 * B_tcols) + c_c_index]);
        B_r6_imag = _mm256_loadu_ps (&B->imagp[(6 * B_tcols) + c_c_index]);
        B_r7_real = _mm256_loadu_ps (&B->realp[(7 * B_tcols) + c_c_index]);
        B_r7_imag = _mm256_loadu_ps (&B->imagp[(7 * B_tcols) + c_c_index]);
        
        for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
        {
            _mm256_storeu_ps (&C->realp[(r_c_index * C_tcols) + c_c_index],
               _mm256_add_ps (
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (B_r0_real, A_c0_real[r_c_index]),
                                                  _mm256_mul_ps (B_r0_imag, A_c0_imag[r_c_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (B_r1_real, A_c1_real[r_c_index]),
                                                  _mm256_mul_ps (B_r1_imag, A_c1_imag[r_c_index]))),
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (B_r2_real, A_c2_real[r_c_index]),
                                                  _mm256_mul_ps (B_r2_imag, A_c2_imag[r_c_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (B_r3_real, A_c3_real[r_c_index]),
                                                  _mm256_mul_ps (B_r3_imag, A_c3_imag[r_c_index])))),
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (B_r4_real, A_c4_real[r_c_index]),
                                                  _mm256_mul_ps (B_r4_imag, A_c4_imag[r_c_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (B_r5_real, A_c5_real[r_c_index]),
                                                  _mm256_mul_ps (B_r5_imag, A_c5_imag[r_c_index]))),
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (B_r6_real, A_c6_real[r_c_index]),
                                                  _mm256_mul_ps (B_r6_imag, A_c6_imag[r_c_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (B_r7_real, A_c7_real[r_c_index]),
                                                  _mm256_mul_ps (B_r7_imag, A_c7_imag[r_c_index]))))));

            _mm256_storeu_ps (&C->imagp[(r_c_index * C_tcols) + c_c_index],
               _mm256_add_ps (
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (B_r0_imag, A_c0_real[r_c_index]),
                                                  _mm256_mul_ps (B_r0_real, A_c0_imag[r_c_index])),
                                   _mm256_add_ps (_mm256_mul_ps (B_r1_imag, A_c1_real[r_c_index]),
                                                  _mm256_mul_ps (B_r1_real, A_c1_imag[r_c_index]))),
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (B_r2_imag, A_c2_real[r_c_index]),
                                                  _mm256_mul_ps (B_r2_real, A_c2_imag[r_c_index])),
                                   _mm256_add_ps (_mm256_mul_ps (B_r3_imag, A_c3_real[r_c_index]),
                                                  _mm256_mul_ps (B_r3_real, A_c3_imag[r_c_index])))),
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (B_r4_imag, A_c4_real[r_c_index]),
                                                  _mm256_mul_ps (B_r4_real, A_c4_imag[r_c_index])),
                                   _mm256_add_ps (_mm256_mul_ps (B_r5_imag, A_c5_real[r_c_index]),
                                                  _mm256_mul_ps (B_r5_real, A_c5_imag[r_c_index]))),
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (B_r6_imag, A_c6_real[r_c_index]),
                                                  _mm256_mul_ps (B_r6_real, A_c6_imag[r_c_index])),
                                   _mm256_add_ps (_mm256_mul_ps (B_r7_imag, A_c7_real[r_c_index]),
                                                  _mm256_mul_ps (B_r7_real, A_c7_imag[r_c_index]))))));
        }
    }
    
    return SAL_SUCCESS;
}

SAL_i32 cmat_mulx_avx_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                        SAL_i32 A_tcols, 	/* left input column stride */
  	                                        SAL_cf32 *B, 	/* right input matrix */
  	                                        SAL_i32 B_tcols, 	/* right input column stride */
  	                                        SAL_cf32 *C, 	/* output matrix */
  	                                        SAL_i32 nr_c, 	/* rows count in C */
                                            SAL_i32 C_tcols, 	/* output column stride */
  	                                        SAL_i32 nc_c) 	/* column count in C */
{
    __m256 A_c0_real[NR_C_MAX], A_c0_imag[NR_C_MAX];
    __m256 A_c1_real[NR_C_MAX], A_c1_imag[NR_C_MAX];
    __m256 A_c2_real[NR_C_MAX], A_c2_imag[NR_C_MAX];
    __m256 A_c3_real[NR_C_MAX], A_c3_imag[NR_C_MAX];
    __m256 A_c4_real[NR_C_MAX], A_c4_imag[NR_C_MAX];
    __m256 A_c5_real[NR_C_MAX], A_c5_imag[NR_C_MAX];
    __m256 A_c6_real[NR_C_MAX], A_c6_imag[NR_C_MAX];
    __m256 A_c7_real[NR_C_MAX], A_c7_imag[NR_C_MAX];
    __m256 B_r0_real_imag, B_r0_imag_real;
    __m256 B_r1_real_imag, B_r1_imag_real;
    __m256 B_r2_real_imag, B_r2_imag_real;
    __m256 B_r3_real_imag, B_r3_imag_real;
    __m256 B_r4_real_imag, B_r4_imag_real;
    __m256 B_r5_real_imag, B_r5_imag_real;
    __m256 B_r6_real_imag, B_r6_imag_real;
    __m256 B_r7_real_imag, B_r7_imag_real;
    SAL_i32 r_c_index, c_c_index;
    
    for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
    {
        A_c0_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 0].real);
        A_c0_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 0].imag);
        A_c1_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 1].real);
        A_c1_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 1].imag);
        A_c2_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 2].real);
        A_c2_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 2].imag);
        A_c3_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 3].real);
        A_c3_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 3].imag);
        A_c4_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 4].real);
        A_c4_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 4].imag);
        A_c5_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 5].real);
        A_c5_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 5].imag);
        A_c6_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 6].real);
        A_c6_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 6].imag);
        A_c7_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 7].real);
        A_c7_imag[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 7].imag);
    }
    
    for (c_c_index = 0; c_c_index < nc_c; c_c_index += 4)
    {
        B_r0_real_imag = _mm256_loadu_ps (&B[(0 * B_tcols) + c_c_index].real);
        B_r0_imag_real = _mm256_permute_ps (B_r0_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r1_real_imag = _mm256_loadu_ps (&B[(1 * B_tcols) + c_c_index].real);
        B_r1_imag_real = _mm256_permute_ps (B_r1_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r2_real_imag = _mm256_loadu_ps (&B[(2 * B_tcols) + c_c_index].real);
        B_r2_imag_real = _mm256_permute_ps (B_r2_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r3_real_imag = _mm256_loadu_ps (&B[(3 * B_tcols) + c_c_index].real);
        B_r3_imag_real = _mm256_permute_ps (B_r3_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r4_real_imag = _mm256_loadu_ps (&B[(4 * B_tcols) + c_c_index].real);
        B_r4_imag_real = _mm256_permute_ps (B_r4_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r5_real_imag = _mm256_loadu_ps (&B[(5 * B_tcols) + c_c_index].real);
        B_r5_imag_real = _mm256_permute_ps (B_r5_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r6_real_imag = _mm256_loadu_ps (&B[(6 * B_tcols) + c_c_index].real);
        B_r6_imag_real = _mm256_permute_ps (B_r6_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r7_real_imag = _mm256_loadu_ps (&B[(7 * B_tcols) + c_c_index].real);
        B_r7_imag_real = _mm256_permute_ps (B_r7_real_imag, SWAP_REAL_IMAG_PERMUTE);
        
        for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
        {
            _mm256_storeu_ps (&C[(r_c_index * C_tcols) + c_c_index].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (B_r0_real_imag, A_c0_real[r_c_index]),
                                                      _mm256_mul_ps (B_r0_imag_real, A_c0_imag[r_c_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (B_r1_real_imag, A_c1_real[r_c_index]),
                                                      _mm256_mul_ps (B_r1_imag_real, A_c1_imag[r_c_index]))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (B_r2_real_imag, A_c2_real[r_c_index]),
                                                      _mm256_mul_ps (B_r2_imag_real, A_c2_imag[r_c_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (B_r3_real_imag, A_c3_real[r_c_index]),
                                                      _mm256_mul_ps (B_r3_imag_real, A_c3_imag[r_c_index])))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (B_r4_real_imag, A_c4_real[r_c_index]),
                                                      _mm256_mul_ps (B_r4_imag_real, A_c4_imag[r_c_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (B_r5_real_imag, A_c5_real[r_c_index]),
                                                      _mm256_mul_ps (B_r5_imag_real, A_c5_imag[r_c_index]))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (B_r6_real_imag, A_c6_real[r_c_index]),
                                                      _mm256_mul_ps (B_r6_imag_real, A_c6_imag[r_c_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (B_r7_real_imag, A_c7_real[r_c_index]),
                                                      _mm256_mul_ps (B_r7_imag_real, A_c7_imag[r_c_index]))))));
        }
    }
    
    return SAL_SUCCESS;
}

SAL_i32 cmat_mulx_avx_fma_dot_product_length_8 (SAL_cf32 *A, 	/* left input matrix */
  	                                            SAL_i32 A_tcols, 	/* left input column stride */
  	                                            SAL_cf32 *B, 	/* right input matrix */
  	                                            SAL_i32 B_tcols, 	/* right input column stride */
  	                                            SAL_cf32 *C, 	/* output matrix */
  	                                            SAL_i32 nr_c, 	/* rows count in C */
                                                SAL_i32 C_tcols, 	/* output column stride */
  	                                            SAL_i32 nc_c) 	/* column count in C */
{
    const __m256 negate_even_sign = _mm256_set_ps (1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    __m256 A_c0_real[NR_C_MAX], A_c0_imag[NR_C_MAX];
    __m256 A_c1_real[NR_C_MAX], A_c1_imag[NR_C_MAX];
    __m256 A_c2_real[NR_C_MAX], A_c2_imag[NR_C_MAX];
    __m256 A_c3_real[NR_C_MAX], A_c3_imag[NR_C_MAX];
    __m256 A_c4_real[NR_C_MAX], A_c4_imag[NR_C_MAX];
    __m256 A_c5_real[NR_C_MAX], A_c5_imag[NR_C_MAX];
    __m256 A_c6_real[NR_C_MAX], A_c6_imag[NR_C_MAX];
    __m256 A_c7_real[NR_C_MAX], A_c7_imag[NR_C_MAX];
    __m256 B_r0_real_imag, B_r0_imag_real;
    __m256 B_r1_real_imag, B_r1_imag_real;
    __m256 B_r2_real_imag, B_r2_imag_real;
    __m256 B_r3_real_imag, B_r3_imag_real;
    __m256 B_r4_real_imag, B_r4_imag_real;
    __m256 B_r5_real_imag, B_r5_imag_real;
    __m256 B_r6_real_imag, B_r6_imag_real;
    __m256 B_r7_real_imag, B_r7_imag_real;
    SAL_i32 r_c_index, c_c_index;
    
    for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
    {
        A_c0_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 0].real);
        A_c0_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 0].imag));
        A_c1_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 1].real);
        A_c1_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 1].imag));
        A_c2_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 2].real);
        A_c2_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 2].imag));
        A_c3_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 3].real);
        A_c3_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 3].imag));
        A_c4_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 4].real);
        A_c4_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 4].imag));
        A_c5_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 5].real);
        A_c5_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 5].imag));
        A_c6_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 6].real);
        A_c6_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 6].imag));
        A_c7_real[r_c_index] = _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 7].real);
        A_c7_imag[r_c_index] = _mm256_mul_ps (negate_even_sign, _mm256_broadcast_ss (&A[(r_c_index * A_tcols) + 7].imag));
    }
    
    for (c_c_index = 0; c_c_index < nc_c; c_c_index += 4)
    {
        B_r0_real_imag = _mm256_loadu_ps (&B[(0 * B_tcols) + c_c_index].real);
        B_r0_imag_real = _mm256_permute_ps (B_r0_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r1_real_imag = _mm256_loadu_ps (&B[(1 * B_tcols) + c_c_index].real);
        B_r1_imag_real = _mm256_permute_ps (B_r1_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r2_real_imag = _mm256_loadu_ps (&B[(2 * B_tcols) + c_c_index].real);
        B_r2_imag_real = _mm256_permute_ps (B_r2_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r3_real_imag = _mm256_loadu_ps (&B[(3 * B_tcols) + c_c_index].real);
        B_r3_imag_real = _mm256_permute_ps (B_r3_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r4_real_imag = _mm256_loadu_ps (&B[(4 * B_tcols) + c_c_index].real);
        B_r4_imag_real = _mm256_permute_ps (B_r4_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r5_real_imag = _mm256_loadu_ps (&B[(5 * B_tcols) + c_c_index].real);
        B_r5_imag_real = _mm256_permute_ps (B_r5_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r6_real_imag = _mm256_loadu_ps (&B[(6 * B_tcols) + c_c_index].real);
        B_r6_imag_real = _mm256_permute_ps (B_r6_real_imag, SWAP_REAL_IMAG_PERMUTE);
        B_r7_real_imag = _mm256_loadu_ps (&B[(7 * B_tcols) + c_c_index].real);
        B_r7_imag_real = _mm256_permute_ps (B_r7_real_imag, SWAP_REAL_IMAG_PERMUTE);
        
        for (r_c_index = 0; r_c_index < nr_c; r_c_index++)
        {
            _mm256_storeu_ps (&C[(r_c_index * C_tcols) + c_c_index].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_fmadd_ps (               B_r0_real_imag, A_c0_real[r_c_index],
                                                     _mm256_mul_ps (B_r0_imag_real, A_c0_imag[r_c_index])),
                                    _mm256_fmadd_ps (               B_r1_real_imag, A_c1_real[r_c_index],
                                                     _mm256_mul_ps (B_r1_imag_real, A_c1_imag[r_c_index]))),
                     _mm256_add_ps (_mm256_fmadd_ps (               B_r2_real_imag, A_c2_real[r_c_index],
                                                     _mm256_mul_ps (B_r2_imag_real, A_c2_imag[r_c_index])),
                                    _mm256_fmadd_ps (               B_r3_real_imag, A_c3_real[r_c_index],
                                                     _mm256_mul_ps (B_r3_imag_real, A_c3_imag[r_c_index])))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_fmadd_ps (               B_r4_real_imag, A_c4_real[r_c_index],
                                                     _mm256_mul_ps (B_r4_imag_real, A_c4_imag[r_c_index])),
                                    _mm256_fmadd_ps (               B_r5_real_imag, A_c5_real[r_c_index],
                                                     _mm256_mul_ps (B_r5_imag_real, A_c5_imag[r_c_index]))),
                     _mm256_add_ps (_mm256_fmadd_ps (               B_r6_real_imag, A_c6_real[r_c_index],
                                                     _mm256_mul_ps (B_r6_imag_real, A_c6_imag[r_c_index])),
                                    _mm256_fmadd_ps (               B_r7_real_imag, A_c7_real[r_c_index],
                                                     _mm256_mul_ps (B_r7_imag_real, A_c7_imag[r_c_index]))))));
        }
    }
    
    return SAL_SUCCESS;
    }
