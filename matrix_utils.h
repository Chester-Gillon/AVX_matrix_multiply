SAL_cf32* copy_mx_to_cf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols);
void copy_cf32_to_mx_matrix (const SAL_cf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix);
void copy_mx_to_zf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols, SAL_zf32 *const C_matrix);
void copy_zf32_to_mx_matrix (const SAL_zf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix);
mxArray *time_matrix_multiply (void (*matrix_func) (void *),
                               void *test_specific_context,
                               SAL_i32 num_timed_iterations);
