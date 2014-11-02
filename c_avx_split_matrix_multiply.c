
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "avx_matrix_multiply_library.h"
#include "matrix_utils.h"

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
        copy_mx_to_zf32_matrix (left_matrix_in, &left_matrix_tcols, &left_matrix);
        copy_mx_to_zf32_matrix (right_matrix_in, &right_matrix_tcols, &right_matrix);
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
        copy_zf32_to_mx_matrix (&output_matrix, output_matrix_tcols, mx_output_matrix);
    
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