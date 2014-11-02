#include <immintrin.h>
#include <sal.h>

#include "cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.h"

/* For a complex interleave AVX vector swap the real and imaginary parts */
#define SWAP_REAL_IMAG_PERMUTE 0xB1
void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_2 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r;
    __m256 left_r0_c0_i, left_r0_c1_i;
    __m256 left_r1_c0_r, left_r1_c1_r;
    __m256 left_r1_c0_i, left_r1_c1_i;
    __m256 left_r2_c0_r, left_r2_c1_r;
    __m256 left_r2_c0_i, left_r2_c1_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    SAL_i32 c_c;

    left_r0_c0_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].real);
    left_r0_c0_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].imag);
    left_r0_c1_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].real);
    left_r0_c1_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].imag);

    left_r1_c0_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].real);
    left_r1_c0_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].imag);
    left_r1_c1_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].real);
    left_r1_c1_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].imag);

    left_r2_c0_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].real);
    left_r2_c0_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].imag);
    left_r2_c1_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].real);
    left_r2_c1_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c1_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c1_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c1_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_3 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    SAL_i32 c_c;

    left_r0_c0_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].real);
    left_r0_c0_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].imag);
    left_r0_c1_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].real);
    left_r0_c1_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].imag);
    left_r0_c2_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].real);
    left_r0_c2_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].imag);

    left_r1_c0_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].real);
    left_r1_c0_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].imag);
    left_r1_c1_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].real);
    left_r1_c1_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].imag);
    left_r1_c2_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].real);
    left_r1_c2_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].imag);

    left_r2_c0_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].real);
    left_r2_c0_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].imag);
    left_r2_c1_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].real);
    left_r2_c1_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].imag);
    left_r2_c2_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].real);
    left_r2_c2_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c2_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c2_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c2_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_4 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
    SAL_i32 c_c;

    left_r0_c0_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].real);
    left_r0_c0_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 0].imag);
    left_r0_c1_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].real);
    left_r0_c1_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 1].imag);
    left_r0_c2_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].real);
    left_r0_c2_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 2].imag);
    left_r0_c3_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 3].real);
    left_r0_c3_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 3].imag);

    left_r1_c0_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].real);
    left_r1_c0_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 0].imag);
    left_r1_c1_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].real);
    left_r1_c1_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 1].imag);
    left_r1_c2_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].real);
    left_r1_c2_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 2].imag);
    left_r1_c3_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 3].real);
    left_r1_c3_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 3].imag);

    left_r2_c0_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].real);
    left_r2_c0_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 0].imag);
    left_r2_c1_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].real);
    left_r2_c1_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 1].imag);
    left_r2_c2_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].real);
    left_r2_c2_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 2].imag);
    left_r2_c3_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 3].real);
    left_r2_c3_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 3].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c3_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c3_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c3_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_5 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c4_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c4_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c4_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_6 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c5_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c5_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c5_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_7 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c6_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c6_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c6_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_8 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c7_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c7_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c7_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_9 (SAL_cf32 *A,     /* left input matrix */
                                                           SAL_i32 A_tcols, /* left input column stride */
                                                           SAL_cf32 *B,     /* right input matrix */
                                                           SAL_i32 B_tcols, /* right input column stride */
                                                           SAL_cf32 *C,     /* output matrix */
                                                           SAL_i32 C_tcols, /* output column stride */
                                                           SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c8_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c8_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c8_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_10 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c9_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c9_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c9_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_11 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c10_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c10_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c10_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_12 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c11_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c11_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c11_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_13 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c12_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c12_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c12_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_14 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c13_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c13_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c13_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_15 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c14_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c14_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c14_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_16 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);
    left_r0_c15_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].real);
    left_r0_c15_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);
    left_r1_c15_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].real);
    left_r1_c15_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);
    left_r2_c15_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].real);
    left_r2_c15_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        right_r_i = _mm256_load_ps (&B[(15 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c15_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c15_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c15_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_17 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);
    left_r0_c15_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].real);
    left_r0_c15_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].imag);
    left_r0_c16_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].real);
    left_r0_c16_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);
    left_r1_c15_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].real);
    left_r1_c15_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].imag);
    left_r1_c16_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].real);
    left_r1_c16_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);
    left_r2_c15_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].real);
    left_r2_c15_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].imag);
    left_r2_c16_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].real);
    left_r2_c16_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        right_r_i = _mm256_load_ps (&B[(15 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c15_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c15_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c15_i)));
        right_r_i = _mm256_load_ps (&B[(16 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c16_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c16_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c16_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c16_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c16_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c16_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_18 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r, left_r0_c17_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i, left_r0_c17_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r, left_r1_c17_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i, left_r1_c17_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r, left_r2_c17_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i, left_r2_c17_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);
    left_r0_c15_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].real);
    left_r0_c15_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].imag);
    left_r0_c16_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].real);
    left_r0_c16_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].imag);
    left_r0_c17_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].real);
    left_r0_c17_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);
    left_r1_c15_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].real);
    left_r1_c15_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].imag);
    left_r1_c16_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].real);
    left_r1_c16_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].imag);
    left_r1_c17_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].real);
    left_r1_c17_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);
    left_r2_c15_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].real);
    left_r2_c15_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].imag);
    left_r2_c16_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].real);
    left_r2_c16_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].imag);
    left_r2_c17_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].real);
    left_r2_c17_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        right_r_i = _mm256_load_ps (&B[(15 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c15_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c15_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c15_i)));
        right_r_i = _mm256_load_ps (&B[(16 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c16_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c16_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c16_i)));
        right_r_i = _mm256_load_ps (&B[(17 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c17_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c17_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c17_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c17_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c17_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c17_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_19 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r, left_r0_c17_r, left_r0_c18_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i, left_r0_c17_i, left_r0_c18_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r, left_r1_c17_r, left_r1_c18_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i, left_r1_c17_i, left_r1_c18_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r, left_r2_c17_r, left_r2_c18_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i, left_r2_c17_i, left_r2_c18_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);
    left_r0_c15_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].real);
    left_r0_c15_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].imag);
    left_r0_c16_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].real);
    left_r0_c16_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].imag);
    left_r0_c17_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].real);
    left_r0_c17_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].imag);
    left_r0_c18_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 18].real);
    left_r0_c18_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 18].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);
    left_r1_c15_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].real);
    left_r1_c15_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].imag);
    left_r1_c16_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].real);
    left_r1_c16_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].imag);
    left_r1_c17_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].real);
    left_r1_c17_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].imag);
    left_r1_c18_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 18].real);
    left_r1_c18_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 18].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);
    left_r2_c15_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].real);
    left_r2_c15_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].imag);
    left_r2_c16_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].real);
    left_r2_c16_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].imag);
    left_r2_c17_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].real);
    left_r2_c17_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].imag);
    left_r2_c18_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 18].real);
    left_r2_c18_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 18].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        right_r_i = _mm256_load_ps (&B[(15 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c15_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c15_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c15_i)));
        right_r_i = _mm256_load_ps (&B[(16 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c16_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c16_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c16_i)));
        right_r_i = _mm256_load_ps (&B[(17 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c17_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c17_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c17_i)));
        right_r_i = _mm256_load_ps (&B[(18 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c18_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c18_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c18_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c18_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c18_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c18_i))));
    }
}

void cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_20 (SAL_cf32 *A,     /* left input matrix */
                                                            SAL_i32 A_tcols, /* left input column stride */
                                                            SAL_cf32 *B,     /* right input matrix */
                                                            SAL_i32 B_tcols, /* right input column stride */
                                                            SAL_cf32 *C,     /* output matrix */
                                                            SAL_i32 C_tcols, /* output column stride */
                                                            SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r0_c0_r, left_r0_c1_r, left_r0_c2_r, left_r0_c3_r, left_r0_c4_r, left_r0_c5_r, left_r0_c6_r, left_r0_c7_r, left_r0_c8_r, left_r0_c9_r, left_r0_c10_r, left_r0_c11_r, left_r0_c12_r, left_r0_c13_r, left_r0_c14_r, left_r0_c15_r, left_r0_c16_r, left_r0_c17_r, left_r0_c18_r, left_r0_c19_r;
    __m256 left_r0_c0_i, left_r0_c1_i, left_r0_c2_i, left_r0_c3_i, left_r0_c4_i, left_r0_c5_i, left_r0_c6_i, left_r0_c7_i, left_r0_c8_i, left_r0_c9_i, left_r0_c10_i, left_r0_c11_i, left_r0_c12_i, left_r0_c13_i, left_r0_c14_i, left_r0_c15_i, left_r0_c16_i, left_r0_c17_i, left_r0_c18_i, left_r0_c19_i;
    __m256 left_r1_c0_r, left_r1_c1_r, left_r1_c2_r, left_r1_c3_r, left_r1_c4_r, left_r1_c5_r, left_r1_c6_r, left_r1_c7_r, left_r1_c8_r, left_r1_c9_r, left_r1_c10_r, left_r1_c11_r, left_r1_c12_r, left_r1_c13_r, left_r1_c14_r, left_r1_c15_r, left_r1_c16_r, left_r1_c17_r, left_r1_c18_r, left_r1_c19_r;
    __m256 left_r1_c0_i, left_r1_c1_i, left_r1_c2_i, left_r1_c3_i, left_r1_c4_i, left_r1_c5_i, left_r1_c6_i, left_r1_c7_i, left_r1_c8_i, left_r1_c9_i, left_r1_c10_i, left_r1_c11_i, left_r1_c12_i, left_r1_c13_i, left_r1_c14_i, left_r1_c15_i, left_r1_c16_i, left_r1_c17_i, left_r1_c18_i, left_r1_c19_i;
    __m256 left_r2_c0_r, left_r2_c1_r, left_r2_c2_r, left_r2_c3_r, left_r2_c4_r, left_r2_c5_r, left_r2_c6_r, left_r2_c7_r, left_r2_c8_r, left_r2_c9_r, left_r2_c10_r, left_r2_c11_r, left_r2_c12_r, left_r2_c13_r, left_r2_c14_r, left_r2_c15_r, left_r2_c16_r, left_r2_c17_r, left_r2_c18_r, left_r2_c19_r;
    __m256 left_r2_c0_i, left_r2_c1_i, left_r2_c2_i, left_r2_c3_i, left_r2_c4_i, left_r2_c5_i, left_r2_c6_i, left_r2_c7_i, left_r2_c8_i, left_r2_c9_i, left_r2_c10_i, left_r2_c11_i, left_r2_c12_i, left_r2_c13_i, left_r2_c14_i, left_r2_c15_i, left_r2_c16_i, left_r2_c17_i, left_r2_c18_i, left_r2_c19_i;
    __m256 right_r_i, right_i_r;
    __m256 output_r0;
    __m256 output_r1;
    __m256 output_r2;
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
    left_r0_c8_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].real);
    left_r0_c8_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 8].imag);
    left_r0_c9_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].real);
    left_r0_c9_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 9].imag);
    left_r0_c10_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].real);
    left_r0_c10_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 10].imag);
    left_r0_c11_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].real);
    left_r0_c11_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 11].imag);
    left_r0_c12_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].real);
    left_r0_c12_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 12].imag);
    left_r0_c13_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].real);
    left_r0_c13_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 13].imag);
    left_r0_c14_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].real);
    left_r0_c14_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 14].imag);
    left_r0_c15_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].real);
    left_r0_c15_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 15].imag);
    left_r0_c16_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].real);
    left_r0_c16_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 16].imag);
    left_r0_c17_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].real);
    left_r0_c17_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 17].imag);
    left_r0_c18_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 18].real);
    left_r0_c18_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 18].imag);
    left_r0_c19_r = _mm256_broadcast_ss (&A[(0 * A_tcols) + 19].real);
    left_r0_c19_i = _mm256_broadcast_ss (&A[(0 * A_tcols) + 19].imag);

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
    left_r1_c8_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].real);
    left_r1_c8_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 8].imag);
    left_r1_c9_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].real);
    left_r1_c9_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 9].imag);
    left_r1_c10_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].real);
    left_r1_c10_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 10].imag);
    left_r1_c11_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].real);
    left_r1_c11_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 11].imag);
    left_r1_c12_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].real);
    left_r1_c12_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 12].imag);
    left_r1_c13_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].real);
    left_r1_c13_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 13].imag);
    left_r1_c14_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].real);
    left_r1_c14_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 14].imag);
    left_r1_c15_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].real);
    left_r1_c15_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 15].imag);
    left_r1_c16_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].real);
    left_r1_c16_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 16].imag);
    left_r1_c17_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].real);
    left_r1_c17_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 17].imag);
    left_r1_c18_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 18].real);
    left_r1_c18_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 18].imag);
    left_r1_c19_r = _mm256_broadcast_ss (&A[(1 * A_tcols) + 19].real);
    left_r1_c19_i = _mm256_broadcast_ss (&A[(1 * A_tcols) + 19].imag);

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
    left_r2_c8_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].real);
    left_r2_c8_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 8].imag);
    left_r2_c9_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].real);
    left_r2_c9_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 9].imag);
    left_r2_c10_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].real);
    left_r2_c10_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 10].imag);
    left_r2_c11_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].real);
    left_r2_c11_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 11].imag);
    left_r2_c12_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].real);
    left_r2_c12_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 12].imag);
    left_r2_c13_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].real);
    left_r2_c13_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 13].imag);
    left_r2_c14_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].real);
    left_r2_c14_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 14].imag);
    left_r2_c15_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].real);
    left_r2_c15_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 15].imag);
    left_r2_c16_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].real);
    left_r2_c16_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 16].imag);
    left_r2_c17_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].real);
    left_r2_c17_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 17].imag);
    left_r2_c18_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 18].real);
    left_r2_c18_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 18].imag);
    left_r2_c19_r = _mm256_broadcast_ss (&A[(2 * A_tcols) + 19].real);
    left_r2_c19_i = _mm256_broadcast_ss (&A[(2 * A_tcols) + 19].imag);

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        right_r_i = _mm256_load_ps (&B[(0 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r0_c0_i));
        output_r1 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r1_c0_i));
        output_r2 = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c0_r),
                                       _mm256_mul_ps (right_i_r, left_r2_c0_i));
        right_r_i = _mm256_load_ps (&B[(1 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c1_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c1_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c1_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c1_i)));
        right_r_i = _mm256_load_ps (&B[(2 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c2_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c2_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c2_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c2_i)));
        right_r_i = _mm256_load_ps (&B[(3 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c3_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c3_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c3_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c3_i)));
        right_r_i = _mm256_load_ps (&B[(4 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c4_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c4_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c4_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c4_i)));
        right_r_i = _mm256_load_ps (&B[(5 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c5_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c5_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c5_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c5_i)));
        right_r_i = _mm256_load_ps (&B[(6 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c6_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c6_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c6_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c6_i)));
        right_r_i = _mm256_load_ps (&B[(7 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c7_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c7_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c7_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c7_i)));
        right_r_i = _mm256_load_ps (&B[(8 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c8_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c8_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c8_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c8_i)));
        right_r_i = _mm256_load_ps (&B[(9 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c9_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c9_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c9_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c9_i)));
        right_r_i = _mm256_load_ps (&B[(10 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c10_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c10_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c10_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c10_i)));
        right_r_i = _mm256_load_ps (&B[(11 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c11_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c11_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c11_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c11_i)));
        right_r_i = _mm256_load_ps (&B[(12 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c12_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c12_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c12_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c12_i)));
        right_r_i = _mm256_load_ps (&B[(13 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c13_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c13_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c13_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c13_i)));
        right_r_i = _mm256_load_ps (&B[(14 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c14_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c14_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c14_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c14_i)));
        right_r_i = _mm256_load_ps (&B[(15 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c15_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c15_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c15_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c15_i)));
        right_r_i = _mm256_load_ps (&B[(16 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c16_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c16_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c16_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c16_i)));
        right_r_i = _mm256_load_ps (&B[(17 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c17_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c17_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c17_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c17_i)));
        right_r_i = _mm256_load_ps (&B[(18 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        output_r0 = _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c18_r),
                                                                  _mm256_mul_ps (right_i_r, left_r0_c18_i)));
        output_r1 = _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c18_r),
                                                                  _mm256_mul_ps (right_i_r, left_r1_c18_i)));
        output_r2 = _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c18_r),
                                                                  _mm256_mul_ps (right_i_r, left_r2_c18_i)));
        right_r_i = _mm256_load_ps (&B[(19 * B_tcols) + c_c].real);
        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
        _mm256_store_ps (&C[(0 * C_tcols) + c_c].real, _mm256_add_ps (output_r0, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r0_c19_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r0_c19_i))));
        _mm256_store_ps (&C[(1 * C_tcols) + c_c].real, _mm256_add_ps (output_r1, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r1_c19_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r1_c19_i))));
        _mm256_store_ps (&C[(2 * C_tcols) + c_c].real, _mm256_add_ps (output_r2, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r2_c19_r),
                                                                                                     _mm256_mul_ps (right_i_r, left_r2_c19_i))));
    }
}


const cmat_mulx_fixed_dimension_accumulate_func cmat_mulx_fixed_dimension_accumulate_functions[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH+1] =
{
    {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    {NULL,NULL,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_2,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_3,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_4,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_5,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_6,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_7,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_8,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_9,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_10,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_11,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_12,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_13,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_14,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_15,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_16,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_17,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_18,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_19,cmat_mulx_avx_accumulate_nr_c_3_dot_product_length_20}
};
