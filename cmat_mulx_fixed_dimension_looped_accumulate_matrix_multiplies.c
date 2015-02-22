#include <immintrin.h>
#include <sal.h>

#include "cmat_mulx_fixed_dimension_looped_accumulate_matrix_multiplies.h"

/* For a complex interleave AVX vector swap the real and imaginary parts */
#define SWAP_REAL_IMAG_PERMUTE 0xB1
void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_2 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][2];
    __m256 left_i[2][2];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_3 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][3];
    __m256 left_i[2][3];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_4 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][4];
    __m256 left_i[2][4];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_5 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][5];
    __m256 left_i[2][5];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_6 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][6];
    __m256 left_i[2][6];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_7 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][7];
    __m256 left_i[2][7];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_8 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][8];
    __m256 left_i[2][8];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_9 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][9];
    __m256 left_i[2][9];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_10 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][10];
    __m256 left_i[2][10];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_11 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][11];
    __m256 left_i[2][11];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_12 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][12];
    __m256 left_i[2][12];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_13 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][13];
    __m256 left_i[2][13];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_14 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][14];
    __m256 left_i[2][14];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_15 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][15];
    __m256 left_i[2][15];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_16 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][16];
    __m256 left_i[2][16];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_17 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][17];
    __m256 left_i[2][17];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_18 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][18];
    __m256 left_i[2][18];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_19 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][19];
    __m256 left_i[2][19];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_20 (SAL_cf32 *A[2],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[2],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[2][20];
    __m256 left_i[2][20];
    __m256 right_r_i, right_i_r;
    __m256 output[2];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 2; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 2; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 2; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 2; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_2 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][2];
    __m256 left_i[3][2];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_3 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][3];
    __m256 left_i[3][3];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_4 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][4];
    __m256 left_i[3][4];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_5 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][5];
    __m256 left_i[3][5];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_6 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][6];
    __m256 left_i[3][6];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_7 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][7];
    __m256 left_i[3][7];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_8 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][8];
    __m256 left_i[3][8];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_9 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][9];
    __m256 left_i[3][9];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_10 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][10];
    __m256 left_i[3][10];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_11 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][11];
    __m256 left_i[3][11];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_12 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][12];
    __m256 left_i[3][12];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_13 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][13];
    __m256 left_i[3][13];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_14 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][14];
    __m256 left_i[3][14];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_15 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][15];
    __m256 left_i[3][15];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_16 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][16];
    __m256 left_i[3][16];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_17 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][17];
    __m256 left_i[3][17];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_18 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][18];
    __m256 left_i[3][18];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_19 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][19];
    __m256 left_i[3][19];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_20 (SAL_cf32 *A[3],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[3],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[3][20];
    __m256 left_i[3][20];
    __m256 right_r_i, right_i_r;
    __m256 output[3];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 3; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 3; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 3; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 3; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_2 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][2];
    __m256 left_i[4][2];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_3 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][3];
    __m256 left_i[4][3];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_4 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][4];
    __m256 left_i[4][4];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_5 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][5];
    __m256 left_i[4][5];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_6 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][6];
    __m256 left_i[4][6];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_7 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][7];
    __m256 left_i[4][7];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_8 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][8];
    __m256 left_i[4][8];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_9 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][9];
    __m256 left_i[4][9];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_10 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][10];
    __m256 left_i[4][10];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_11 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][11];
    __m256 left_i[4][11];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_12 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][12];
    __m256 left_i[4][12];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_13 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][13];
    __m256 left_i[4][13];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_14 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][14];
    __m256 left_i[4][14];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_15 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][15];
    __m256 left_i[4][15];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_16 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][16];
    __m256 left_i[4][16];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_17 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][17];
    __m256 left_i[4][17];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_18 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][18];
    __m256 left_i[4][18];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_19 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][19];
    __m256 left_i[4][19];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_20 (SAL_cf32 *A[4],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[4],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[4][20];
    __m256 left_i[4][20];
    __m256 right_r_i, right_i_r;
    __m256 output[4];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 4; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 4; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 4; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 4; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_2 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][2];
    __m256 left_i[5][2];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_3 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][3];
    __m256 left_i[5][3];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_4 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][4];
    __m256 left_i[5][4];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_5 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][5];
    __m256 left_i[5][5];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_6 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][6];
    __m256 left_i[5][6];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_7 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][7];
    __m256 left_i[5][7];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_8 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][8];
    __m256 left_i[5][8];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_9 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][9];
    __m256 left_i[5][9];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_10 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][10];
    __m256 left_i[5][10];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_11 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][11];
    __m256 left_i[5][11];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_12 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][12];
    __m256 left_i[5][12];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_13 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][13];
    __m256 left_i[5][13];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_14 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][14];
    __m256 left_i[5][14];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_15 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][15];
    __m256 left_i[5][15];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_16 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][16];
    __m256 left_i[5][16];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_17 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][17];
    __m256 left_i[5][17];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_18 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][18];
    __m256 left_i[5][18];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_19 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][19];
    __m256 left_i[5][19];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_20 (SAL_cf32 *A[5],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[5],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[5][20];
    __m256 left_i[5][20];
    __m256 right_r_i, right_i_r;
    __m256 output[5];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 5; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 5; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 5; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 5; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_2 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][2];
    __m256 left_i[6][2];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_3 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][3];
    __m256 left_i[6][3];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_4 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][4];
    __m256 left_i[6][4];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_5 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][5];
    __m256 left_i[6][5];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_6 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][6];
    __m256 left_i[6][6];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_7 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][7];
    __m256 left_i[6][7];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_8 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][8];
    __m256 left_i[6][8];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_9 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][9];
    __m256 left_i[6][9];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_10 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][10];
    __m256 left_i[6][10];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_11 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][11];
    __m256 left_i[6][11];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_12 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][12];
    __m256 left_i[6][12];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_13 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][13];
    __m256 left_i[6][13];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_14 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][14];
    __m256 left_i[6][14];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_15 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][15];
    __m256 left_i[6][15];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_16 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][16];
    __m256 left_i[6][16];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_17 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][17];
    __m256 left_i[6][17];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_18 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][18];
    __m256 left_i[6][18];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_19 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][19];
    __m256 left_i[6][19];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_20 (SAL_cf32 *A[6],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[6],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[6][20];
    __m256 left_i[6][20];
    __m256 right_r_i, right_i_r;
    __m256 output[6];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 6; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 6; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 6; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 6; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_2 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][2];
    __m256 left_i[7][2];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_3 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][3];
    __m256 left_i[7][3];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_4 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][4];
    __m256 left_i[7][4];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_5 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][5];
    __m256 left_i[7][5];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_6 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][6];
    __m256 left_i[7][6];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_7 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][7];
    __m256 left_i[7][7];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_8 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][8];
    __m256 left_i[7][8];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_9 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][9];
    __m256 left_i[7][9];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_10 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][10];
    __m256 left_i[7][10];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_11 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][11];
    __m256 left_i[7][11];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_12 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][12];
    __m256 left_i[7][12];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_13 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][13];
    __m256 left_i[7][13];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_14 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][14];
    __m256 left_i[7][14];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_15 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][15];
    __m256 left_i[7][15];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_16 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][16];
    __m256 left_i[7][16];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_17 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][17];
    __m256 left_i[7][17];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_18 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][18];
    __m256 left_i[7][18];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_19 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][19];
    __m256 left_i[7][19];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_20 (SAL_cf32 *A[7],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[7],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[7][20];
    __m256 left_i[7][20];
    __m256 right_r_i, right_i_r;
    __m256 output[7];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 7; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 7; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 7; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 7; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_2 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][2];
    __m256 left_i[8][2];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_3 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][3];
    __m256 left_i[8][3];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_4 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][4];
    __m256 left_i[8][4];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_5 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][5];
    __m256 left_i[8][5];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_6 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][6];
    __m256 left_i[8][6];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_7 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][7];
    __m256 left_i[8][7];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_8 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][8];
    __m256 left_i[8][8];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_9 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][9];
    __m256 left_i[8][9];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_10 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][10];
    __m256 left_i[8][10];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_11 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][11];
    __m256 left_i[8][11];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_12 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][12];
    __m256 left_i[8][12];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_13 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][13];
    __m256 left_i[8][13];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_14 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][14];
    __m256 left_i[8][14];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_15 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][15];
    __m256 left_i[8][15];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_16 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][16];
    __m256 left_i[8][16];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_17 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][17];
    __m256 left_i[8][17];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_18 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][18];
    __m256 left_i[8][18];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_19 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][19];
    __m256 left_i[8][19];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_20 (SAL_cf32 *A[8],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[8],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[8][20];
    __m256 left_i[8][20];
    __m256 right_r_i, right_i_r;
    __m256 output[8];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 8; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 8; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 8; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 8; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_2 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][2];
    __m256 left_i[9][2];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_3 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][3];
    __m256 left_i[9][3];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_4 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][4];
    __m256 left_i[9][4];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_5 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][5];
    __m256 left_i[9][5];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_6 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][6];
    __m256 left_i[9][6];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_7 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][7];
    __m256 left_i[9][7];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_8 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][8];
    __m256 left_i[9][8];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_9 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                  SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                  SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                  SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][9];
    __m256 left_i[9][9];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_10 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][10];
    __m256 left_i[9][10];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_11 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][11];
    __m256 left_i[9][11];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_12 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][12];
    __m256 left_i[9][12];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_13 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][13];
    __m256 left_i[9][13];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_14 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][14];
    __m256 left_i[9][14];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_15 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][15];
    __m256 left_i[9][15];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_16 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][16];
    __m256 left_i[9][16];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_17 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][17];
    __m256 left_i[9][17];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_18 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][18];
    __m256 left_i[9][18];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_19 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][19];
    __m256 left_i[9][19];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_20 (SAL_cf32 *A[9],  /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[9],  /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[9][20];
    __m256 left_i[9][20];
    __m256 right_r_i, right_i_r;
    __m256 output[9];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 9; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 9; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 9; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 9; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_2 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][2];
    __m256 left_i[10][2];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_3 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][3];
    __m256 left_i[10][3];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_4 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][4];
    __m256 left_i[10][4];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_5 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][5];
    __m256 left_i[10][5];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_6 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][6];
    __m256 left_i[10][6];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_7 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][7];
    __m256 left_i[10][7];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_8 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][8];
    __m256 left_i[10][8];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_9 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][9];
    __m256 left_i[10][9];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_10 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][10];
    __m256 left_i[10][10];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_11 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][11];
    __m256 left_i[10][11];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_12 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][12];
    __m256 left_i[10][12];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_13 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][13];
    __m256 left_i[10][13];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_14 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][14];
    __m256 left_i[10][14];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_15 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][15];
    __m256 left_i[10][15];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_16 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][16];
    __m256 left_i[10][16];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_17 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][17];
    __m256 left_i[10][17];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_18 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][18];
    __m256 left_i[10][18];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_19 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][19];
    __m256 left_i[10][19];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_20 (SAL_cf32 *A[10], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[10], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[10][20];
    __m256 left_i[10][20];
    __m256 right_r_i, right_i_r;
    __m256 output[10];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 10; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 10; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 10; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 10; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_2 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][2];
    __m256 left_i[11][2];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_3 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][3];
    __m256 left_i[11][3];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_4 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][4];
    __m256 left_i[11][4];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_5 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][5];
    __m256 left_i[11][5];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_6 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][6];
    __m256 left_i[11][6];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_7 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][7];
    __m256 left_i[11][7];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_8 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][8];
    __m256 left_i[11][8];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_9 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][9];
    __m256 left_i[11][9];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_10 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][10];
    __m256 left_i[11][10];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_11 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][11];
    __m256 left_i[11][11];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_12 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][12];
    __m256 left_i[11][12];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_13 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][13];
    __m256 left_i[11][13];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_14 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][14];
    __m256 left_i[11][14];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_15 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][15];
    __m256 left_i[11][15];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_16 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][16];
    __m256 left_i[11][16];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_17 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][17];
    __m256 left_i[11][17];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_18 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][18];
    __m256 left_i[11][18];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_19 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][19];
    __m256 left_i[11][19];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_20 (SAL_cf32 *A[11], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[11], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[11][20];
    __m256 left_i[11][20];
    __m256 right_r_i, right_i_r;
    __m256 output[11];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 11; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 11; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 11; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 11; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_2 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][2];
    __m256 left_i[12][2];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_3 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][3];
    __m256 left_i[12][3];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_4 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][4];
    __m256 left_i[12][4];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_5 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][5];
    __m256 left_i[12][5];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_6 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][6];
    __m256 left_i[12][6];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_7 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][7];
    __m256 left_i[12][7];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_8 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][8];
    __m256 left_i[12][8];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_9 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][9];
    __m256 left_i[12][9];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_10 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][10];
    __m256 left_i[12][10];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_11 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][11];
    __m256 left_i[12][11];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_12 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][12];
    __m256 left_i[12][12];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_13 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][13];
    __m256 left_i[12][13];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_14 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][14];
    __m256 left_i[12][14];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_15 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][15];
    __m256 left_i[12][15];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_16 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][16];
    __m256 left_i[12][16];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_17 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][17];
    __m256 left_i[12][17];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_18 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][18];
    __m256 left_i[12][18];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_19 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][19];
    __m256 left_i[12][19];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_20 (SAL_cf32 *A[12], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[12], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[12][20];
    __m256 left_i[12][20];
    __m256 right_r_i, right_i_r;
    __m256 output[12];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 12; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 12; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 12; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 12; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_2 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][2];
    __m256 left_i[13][2];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_3 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][3];
    __m256 left_i[13][3];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_4 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][4];
    __m256 left_i[13][4];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_5 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][5];
    __m256 left_i[13][5];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_6 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][6];
    __m256 left_i[13][6];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_7 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][7];
    __m256 left_i[13][7];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_8 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][8];
    __m256 left_i[13][8];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_9 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][9];
    __m256 left_i[13][9];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_10 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][10];
    __m256 left_i[13][10];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_11 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][11];
    __m256 left_i[13][11];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_12 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][12];
    __m256 left_i[13][12];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_13 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][13];
    __m256 left_i[13][13];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_14 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][14];
    __m256 left_i[13][14];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_15 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][15];
    __m256 left_i[13][15];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_16 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][16];
    __m256 left_i[13][16];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_17 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][17];
    __m256 left_i[13][17];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_18 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][18];
    __m256 left_i[13][18];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_19 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][19];
    __m256 left_i[13][19];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_20 (SAL_cf32 *A[13], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[13], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[13][20];
    __m256 left_i[13][20];
    __m256 right_r_i, right_i_r;
    __m256 output[13];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 13; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 13; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 13; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 13; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_2 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][2];
    __m256 left_i[14][2];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_3 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][3];
    __m256 left_i[14][3];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_4 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][4];
    __m256 left_i[14][4];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_5 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][5];
    __m256 left_i[14][5];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_6 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][6];
    __m256 left_i[14][6];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_7 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][7];
    __m256 left_i[14][7];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_8 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][8];
    __m256 left_i[14][8];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_9 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][9];
    __m256 left_i[14][9];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_10 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][10];
    __m256 left_i[14][10];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_11 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][11];
    __m256 left_i[14][11];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_12 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][12];
    __m256 left_i[14][12];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_13 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][13];
    __m256 left_i[14][13];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_14 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][14];
    __m256 left_i[14][14];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_15 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][15];
    __m256 left_i[14][15];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_16 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][16];
    __m256 left_i[14][16];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_17 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][17];
    __m256 left_i[14][17];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_18 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][18];
    __m256 left_i[14][18];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_19 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][19];
    __m256 left_i[14][19];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_20 (SAL_cf32 *A[14], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[14], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[14][20];
    __m256 left_i[14][20];
    __m256 right_r_i, right_i_r;
    __m256 output[14];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 14; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 14; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 14; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_2 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][2];
    __m256 left_i[15][2];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_3 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][3];
    __m256 left_i[15][3];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_4 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][4];
    __m256 left_i[15][4];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_5 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][5];
    __m256 left_i[15][5];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_6 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][6];
    __m256 left_i[15][6];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_7 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][7];
    __m256 left_i[15][7];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_8 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][8];
    __m256 left_i[15][8];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_9 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][9];
    __m256 left_i[15][9];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_10 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][10];
    __m256 left_i[15][10];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_11 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][11];
    __m256 left_i[15][11];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_12 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][12];
    __m256 left_i[15][12];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_13 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][13];
    __m256 left_i[15][13];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_14 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][14];
    __m256 left_i[15][14];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_15 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][15];
    __m256 left_i[15][15];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_16 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][16];
    __m256 left_i[15][16];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_17 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][17];
    __m256 left_i[15][17];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_18 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][18];
    __m256 left_i[15][18];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_19 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][19];
    __m256 left_i[15][19];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_20 (SAL_cf32 *A[15], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[15], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[15][20];
    __m256 left_i[15][20];
    __m256 right_r_i, right_i_r;
    __m256 output[15];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 15; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 15; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 15; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 15; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_2 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][2];
    __m256 left_i[16][2];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_3 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][3];
    __m256 left_i[16][3];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_4 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][4];
    __m256 left_i[16][4];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_5 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][5];
    __m256 left_i[16][5];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_6 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][6];
    __m256 left_i[16][6];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_7 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][7];
    __m256 left_i[16][7];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_8 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][8];
    __m256 left_i[16][8];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_9 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][9];
    __m256 left_i[16][9];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_10 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][10];
    __m256 left_i[16][10];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_11 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][11];
    __m256 left_i[16][11];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_12 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][12];
    __m256 left_i[16][12];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_13 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][13];
    __m256 left_i[16][13];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_14 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][14];
    __m256 left_i[16][14];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_15 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][15];
    __m256 left_i[16][15];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_16 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][16];
    __m256 left_i[16][16];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_17 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][17];
    __m256 left_i[16][17];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_18 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][18];
    __m256 left_i[16][18];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_19 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][19];
    __m256 left_i[16][19];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_20 (SAL_cf32 *A[16], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[16], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[16][20];
    __m256 left_i[16][20];
    __m256 right_r_i, right_i_r;
    __m256 output[16];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 16; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 16; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 16; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 16; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_2 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][2];
    __m256 left_i[17][2];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_3 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][3];
    __m256 left_i[17][3];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_4 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][4];
    __m256 left_i[17][4];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_5 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][5];
    __m256 left_i[17][5];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_6 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][6];
    __m256 left_i[17][6];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_7 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][7];
    __m256 left_i[17][7];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_8 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][8];
    __m256 left_i[17][8];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_9 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][9];
    __m256 left_i[17][9];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_10 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][10];
    __m256 left_i[17][10];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_11 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][11];
    __m256 left_i[17][11];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_12 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][12];
    __m256 left_i[17][12];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_13 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][13];
    __m256 left_i[17][13];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_14 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][14];
    __m256 left_i[17][14];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_15 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][15];
    __m256 left_i[17][15];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_16 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][16];
    __m256 left_i[17][16];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_17 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][17];
    __m256 left_i[17][17];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_18 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][18];
    __m256 left_i[17][18];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_19 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][19];
    __m256 left_i[17][19];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_20 (SAL_cf32 *A[17], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[17], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[17][20];
    __m256 left_i[17][20];
    __m256 right_r_i, right_i_r;
    __m256 output[17];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 17; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 17; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 17; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 17; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_2 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][2];
    __m256 left_i[18][2];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_3 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][3];
    __m256 left_i[18][3];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_4 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][4];
    __m256 left_i[18][4];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_5 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][5];
    __m256 left_i[18][5];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_6 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][6];
    __m256 left_i[18][6];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_7 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][7];
    __m256 left_i[18][7];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_8 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][8];
    __m256 left_i[18][8];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_9 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][9];
    __m256 left_i[18][9];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_10 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][10];
    __m256 left_i[18][10];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_11 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][11];
    __m256 left_i[18][11];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_12 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][12];
    __m256 left_i[18][12];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_13 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][13];
    __m256 left_i[18][13];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_14 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][14];
    __m256 left_i[18][14];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_15 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][15];
    __m256 left_i[18][15];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_16 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][16];
    __m256 left_i[18][16];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_17 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][17];
    __m256 left_i[18][17];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_18 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][18];
    __m256 left_i[18][18];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_19 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][19];
    __m256 left_i[18][19];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_20 (SAL_cf32 *A[18], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[18], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[18][20];
    __m256 left_i[18][20];
    __m256 right_r_i, right_i_r;
    __m256 output[18];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 18; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 18; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 18; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 18; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_2 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][2];
    __m256 left_i[19][2];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_3 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][3];
    __m256 left_i[19][3];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_4 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][4];
    __m256 left_i[19][4];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_5 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][5];
    __m256 left_i[19][5];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_6 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][6];
    __m256 left_i[19][6];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_7 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][7];
    __m256 left_i[19][7];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_8 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][8];
    __m256 left_i[19][8];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_9 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][9];
    __m256 left_i[19][9];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_10 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][10];
    __m256 left_i[19][10];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_11 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][11];
    __m256 left_i[19][11];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_12 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][12];
    __m256 left_i[19][12];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_13 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][13];
    __m256 left_i[19][13];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_14 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][14];
    __m256 left_i[19][14];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_15 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][15];
    __m256 left_i[19][15];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_16 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][16];
    __m256 left_i[19][16];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_17 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][17];
    __m256 left_i[19][17];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_18 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][18];
    __m256 left_i[19][18];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_19 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][19];
    __m256 left_i[19][19];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_20 (SAL_cf32 *A[19], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[19], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[19][20];
    __m256 left_i[19][20];
    __m256 right_r_i, right_i_r;
    __m256 output[19];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 19; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 19; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 19; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 19; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_2 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[2],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][2];
    __m256 left_i[20][2];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 2; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 2; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_3 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[3],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][3];
    __m256 left_i[20][3];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 3; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 3; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_4 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[4],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][4];
    __m256 left_i[20][4];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 4; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 4; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_5 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[5],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][5];
    __m256 left_i[20][5];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 5; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 5; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_6 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[6],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][6];
    __m256 left_i[20][6];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 6; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 6; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_7 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[7],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][7];
    __m256 left_i[20][7];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 7; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 7; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_8 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[8],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][8];
    __m256 left_i[20][8];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 8; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 8; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_9 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                   SAL_cf32 *B[9],  /* right input matrix array [dot_product_length] of pointers to each row */
                                                                   SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                   SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][9];
    __m256 left_i[20][9];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 9; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 9; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_10 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[10], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][10];
    __m256 left_i[20][10];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 10; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 10; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_11 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[11], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][11];
    __m256 left_i[20][11];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 11; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 11; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_12 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[12], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][12];
    __m256 left_i[20][12];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 12; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 12; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_13 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[13], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][13];
    __m256 left_i[20][13];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 13; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 13; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_14 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[14], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][14];
    __m256 left_i[20][14];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 14; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 14; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_15 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[15], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][15];
    __m256 left_i[20][15];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 15; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 15; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_16 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[16], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][16];
    __m256 left_i[20][16];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 16; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 16; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_17 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[17], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][17];
    __m256 left_i[20][17];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 17; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 17; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_18 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[18], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][18];
    __m256 left_i[20][18];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 18; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_19 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[19], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][19];
    __m256 left_i[20][19];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 19; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 19; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}

void cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_20 (SAL_cf32 *A[20], /* left input matrix array [nr_c] of pointers to each row */
                                                                    SAL_cf32 *B[20], /* right input matrix array [dot_product_length] of pointers to each row */
                                                                    SAL_cf32 *C[20], /* output matrix array [nr_c] of pointers to each row */
                                                                    SAL_i32 nc_c)    /* column count in C */
{
    __m256 left_r[20][20];
    __m256 left_i[20][20];
    __m256 right_r_i, right_i_r;
    __m256 output[20];
    SAL_i32 c_c;
    SAL_i32 left_row;
    SAL_i32 left_col;
    SAL_i32 output_row;
    SAL_i32 right_row;
    for (left_row = 0; left_row < 20; left_row++)
    {
        for (left_col = 0; left_col < 20; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
        for (output_row = 0; output_row < 20; output_row++)
        {
            output[output_row] = _mm256_set1_ps (0.0f);
        }

        for (right_row = 0; right_row < 20; right_row++)
        {
            right_r_i = _mm256_load_ps (&(B[right_row] + c_c)->real);
            right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);
            for (output_row = 0; output_row < 20; output_row++)
            {
                output[output_row] = _mm256_add_ps (output[output_row], _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r[output_row][right_row]),
                                                                                          _mm256_mul_ps (right_i_r, left_i[output_row][right_row])));
            }
        }

        for (output_row = 0; output_row < 20; output_row++)
        {
            _mm256_store_ps (&(C[output_row] + c_c)->real, output[output_row]);
        }
    }
}


const cmat_mulx_fixed_dimension_looped_accumulate_func cmat_mulx_fixed_dimension_looped_accumulate_functions[CMAT_MULX_FIXED_DIMENSION_LOOPED_ACCUMULATE_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_LOOPED_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH+1] =
{
    {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_2_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_3_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_4_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_5_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_6_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_7_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_8_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_9_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_10_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_11_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_12_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_13_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_14_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_15_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_16_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_17_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_18_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_19_dot_product_length_20},
    {NULL,NULL,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_2,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_3,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_4,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_5,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_6,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_7,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_8,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_9,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_10,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_11,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_12,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_13,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_14,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_15,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_16,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_17,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_18,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_19,cmat_mulx_avx_looped_accumulate_nr_c_20_dot_product_length_20}
};
