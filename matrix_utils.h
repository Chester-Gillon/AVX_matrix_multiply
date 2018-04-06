/** Used to allocate storage for a matrix of distributed rows,
 *  abstracting the underlying memory allocation and alignment of each row */
typedef struct
{
    /* The number of rows in the matrix */
    mwSize num_rows;
    /* The number of usable columns in the matrix */
    mwSize num_cols;
    /* The stride for each row in the matrix, which is num_cols aligned to at least
     * an AVX vector */
    SAL_i32 tcols;
    /* Points to the start of the allocated data for the matrix */
    void *data;
} matrix_storage;

void allocate_cf32_matrix (const mwSize num_rows, const mwSize num_cols, 
                           matrix_storage *const matrix, SAL_cf32 **const matrix_rows);
void allocate_zf32_matrix (const mwSize num_rows, const mwSize num_cols,
                           matrix_storage *const matrix, SAL_zf32 *const matrix_rows);
void free_matrix (matrix_storage *const matrix);
void copy_mx_to_cf32_matrix (const mxArray *const mx_matrix, 
                             matrix_storage *const matrix, SAL_cf32 **const matrix_rows);
mxArray *copy_cf32_to_mx_matrix (const matrix_storage *const matrix);
void copy_mx_to_zf32_matrix (const mxArray *const mx_matrix,
                             matrix_storage *const matrix, SAL_zf32 *const matrix_rows);
mxArray *copy_zf32_to_mx_matrix (const matrix_storage *const matrix);
mxArray *time_matrix_multiply (void (*matrix_func) (void *),
                               void *test_specific_context,
                               const SAL_i32 num_timed_iterations,
                               const bool block_other_cpus);
