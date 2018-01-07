/* For a complex interleave AVX vector swap the real and imaginary parts */
#ifndef SWAP_REAL_IMAG_PERMUTE
#define SWAP_REAL_IMAG_PERMUTE 0xB1
#endif

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
#ifdef IACA_LOAD_LEFT
    IACA_START
#endif
    for (left_row = 0; left_row < 14; left_row++)
    {
        for (left_col = 0; left_col < 18; left_col++)
        {
            left_r[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->real);
            left_i[left_row][left_col] = _mm256_broadcast_ss (&(A[left_row] + left_col)->imag);
        }
    }
#ifdef IACA_LOAD_LEFT
    IACA_END
#endif

    for (c_c = 0; c_c < nc_c; c_c += 4)
    {
#ifdef IACA_OPERATE
        IACA_START
#endif
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
#ifdef IACA_OPERATE
    IACA_END
#endif
}

