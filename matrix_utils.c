#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

SAL_cf32* copy_mx_to_cf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols)
{
    const float *real_in = mxGetData (mx_matrix);
    const float *imag_in = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    SAL_cf32 *C_matrix;
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    /* Align each C matrix column to start on an AVX aligned boundrary */
    *tcols = (num_cols + 3) & ~3;
    C_matrix = mxCalloc (num_rows * *tcols, sizeof(SAL_cf32));
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * *tcols) + col;
            col_major_index = (col * num_rows) + row;
            C_matrix[row_major_index].real = real_in[col_major_index];
            C_matrix[row_major_index].imag = imag_in[col_major_index];
        }
    }
    
    return C_matrix;
}

void copy_cf32_to_mx_matrix (const SAL_cf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix)
{
    float *real_out = mxGetData (mx_matrix);
    float *imag_out = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * tcols) + col;
            col_major_index = (col * num_rows) + row;
            real_out[col_major_index] = C_matrix[row_major_index].real;
            imag_out[col_major_index] = C_matrix[row_major_index].imag;
        }
    }
}

void copy_mx_to_zf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols, SAL_zf32 *const C_matrix)
{
    const float *real_in = mxGetData (mx_matrix);
    const float *imag_in = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    /* Align each C matrix column to start on an AVX aligned boundrary */
    *tcols = (num_cols + 7) & ~7;
    C_matrix->realp = mxCalloc (num_rows * *tcols, sizeof(float));
    C_matrix->imagp = mxCalloc (num_rows * *tcols, sizeof(float));
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * *tcols) + col;
            col_major_index = (col * num_rows) + row;
            C_matrix->realp[row_major_index] = real_in[col_major_index];
            C_matrix->imagp[row_major_index] = imag_in[col_major_index];
        }
    }
}

void copy_zf32_to_mx_matrix (const SAL_zf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix)
{
    float *real_out = mxGetData (mx_matrix);
    float *imag_out = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * tcols) + col;
            col_major_index = (col * num_rows) + row;
            real_out[col_major_index] = C_matrix->realp[row_major_index];
            imag_out[col_major_index] = C_matrix->imagp[row_major_index];
        }
    }
}
