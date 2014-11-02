
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const left_matrix_in = prhs[0];
    const mxArray *const right_matrix_in = prhs[1];
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    SAL_i32 nr_c, nc_c, dot_product_length;
    mxArray *mx_output_matrix;
    SAL_cf32 *left_matrix;
    SAL_cf32 *right_matrix;
    SAL_cf32 *output_matrix;
    SAL_i32 rc;
    SAL_i32 left_matrix_tcols, right_matrix_tcols;

    if (nlhs != 1)
    {
        mexErrMsgIdAndTxt ("c_opensal_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 2)
    {
        mexErrMsgIdAndTxt ("c_opensal_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (left_matrix_in) || !mxIsSingle (left_matrix_in) ||
        !mxIsComplex (right_matrix_in) || !mxIsSingle (right_matrix_in) ||
        (mxGetNumberOfDimensions (left_matrix_in) != 2) ||
        (mxGetNumberOfDimensions (right_matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_opensal_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    nr_c = left_matrix_dimensions[0];
    dot_product_length = left_matrix_dimensions[1];
    nc_c = right_matrix_dimensions[1];
    if (dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:d", "Inconsistent number of weights");
    }

    left_matrix = copy_mx_to_cf32_matrix (left_matrix_in, &left_matrix_tcols);
    right_matrix = copy_mx_to_cf32_matrix (right_matrix_in, &right_matrix_tcols);
    output_matrix = mxCalloc (nr_c * nc_c, sizeof(SAL_cf32));
    
    rc = cmat_mulx (left_matrix, left_matrix_tcols,
                    right_matrix, right_matrix_tcols,
                    output_matrix, nc_c,
                    nr_c, nc_c, dot_product_length, 0, 0);
    if (rc != SAL_SUCCESS)
    {
        mexErrMsgIdAndTxt ("c_opensal_matrix_multiply:c", "cmat_mulx failed with rc=%u", rc);
    }

    mx_output_matrix = mxCreateNumericMatrix (nr_c, nc_c, mxSINGLE_CLASS, mxCOMPLEX);
    plhs[0] = mx_output_matrix;
    copy_cf32_to_mx_matrix (output_matrix, nc_c, mx_output_matrix);
    
    mxFree (left_matrix);
    mxFree (right_matrix);
    mxFree (output_matrix);
}