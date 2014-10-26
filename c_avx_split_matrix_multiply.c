
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "avx_matrix_multiply_library.h"

static void copy_mx_to_C_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols, SAL_zf32 *const C_matrix)
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

static void copy_C_to_mx_matrix (const SAL_zf32 *const C_matrix, const SAL_i32 tcols,
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const left_matrix_in = prhs[0];
    const mxArray *const right_matrix_in = prhs[1];
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    SAL_i32 nr_c, nc_c, dot_product_length;
    mxArray *mx_output_matrix;
    SAL_zf32 left_matrix;
    SAL_zf32 right_matrix;
    SAL_zf32 output_matrix;
    SAL_i32 left_matrix_tcols, right_matrix_tcols, output_matrix_tcols;
    SAL_i32 rc;

    if (nlhs != 1)
    {
        mexErrMsgIdAndTxt ("c_avx_split_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 2)
    {
        mexErrMsgIdAndTxt ("c_avx_split_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (left_matrix_in) || !mxIsSingle (left_matrix_in) ||
        !mxIsComplex (right_matrix_in) || !mxIsSingle (right_matrix_in) ||
        (mxGetNumberOfDimensions (left_matrix_in) != 2) ||
        (mxGetNumberOfDimensions (right_matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_avx_split_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    nr_c = left_matrix_dimensions[0];
    dot_product_length = left_matrix_dimensions[1];
    nc_c = right_matrix_dimensions[1];
    if (dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_avx_split_matrix_multiply:d", "Inconsistent number of weights");
    }
    
    if ((nr_c <= NR_C_MAX) && (dot_product_length == 8))
    {
        copy_mx_to_C_matrix (left_matrix_in, &left_matrix_tcols, &left_matrix);
        copy_mx_to_C_matrix (right_matrix_in, &right_matrix_tcols, &right_matrix);
        output_matrix_tcols = (nc_c + 7) & ~7;
        output_matrix.realp = mxCalloc (nr_c * output_matrix_tcols, sizeof(float));
        output_matrix.imagp = mxCalloc (nr_c * output_matrix_tcols, sizeof(float));

        rc = zmat_mulx_avx_dot_product_length_8 (&left_matrix, left_matrix_tcols,
                                                 &right_matrix, right_matrix_tcols,
                                                 &output_matrix, nr_c, output_matrix_tcols,
                                                 nc_c);
        if (rc != SAL_SUCCESS)
        {
            mexErrMsgIdAndTxt ("c_avx_split_matrix_multiply:e", "zmat_mulx failed with rc=%u", rc);
        }

        mx_output_matrix = mxCreateNumericMatrix (nr_c, nc_c, mxSINGLE_CLASS, mxCOMPLEX);
        copy_C_to_mx_matrix (&output_matrix, output_matrix_tcols, mx_output_matrix);
    
        mxFree (left_matrix.realp);
        mxFree (left_matrix.imagp);
        mxFree (right_matrix.realp);
        mxFree (right_matrix.imagp);
        mxFree (output_matrix.realp);
        mxFree (output_matrix.imagp);
    }
    else
    {
        /* Empty output to indicate the maxtrix dimension isn't support */
        mx_output_matrix = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxCOMPLEX);
    }
    plhs[0] = mx_output_matrix;
}