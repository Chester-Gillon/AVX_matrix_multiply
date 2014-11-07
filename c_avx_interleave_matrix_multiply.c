
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "avx_matrix_multiply_library.h"
#include "matrix_utils.h"

typedef struct
{
    SAL_i32 nr_c, nc_c, dot_product_length;
    SAL_cf32 *left_matrix;
    SAL_cf32 *right_matrix;
    SAL_cf32 *output_matrix;
    SAL_i32 left_matrix_tcols, right_matrix_tcols, output_matrix_tcols;
    SAL_i32 rc;
} matrix_context;

static void timed_c_matrix_multiply (void *arg)
{
    matrix_context *const context = (matrix_context *) arg;

    context->rc = cmat_mulx_avx_dot_product_length_8 (context->left_matrix, context->left_matrix_tcols,
                                                      context->right_matrix, context->right_matrix_tcols,
                                                      context->output_matrix, context->nr_c, context->output_matrix_tcols,
                                                      context->nc_c);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const left_matrix_in = prhs[0];
    const mxArray *const right_matrix_in = prhs[1];
    const mxArray *const num_timed_iterations_in = prhs[2];
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    mxArray *mx_output_matrix;
    mxArray *timing_results;
    matrix_context context;

    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt ("c_avx_interleave_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 3)
    {
        mexErrMsgIdAndTxt ("c_avx_interleave_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (left_matrix_in) || !mxIsSingle (left_matrix_in) ||
        !mxIsComplex (right_matrix_in) || !mxIsSingle (right_matrix_in) ||
        (mxGetNumberOfDimensions (left_matrix_in) != 2) ||
        (mxGetNumberOfDimensions (right_matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_avx_interleave_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    context.nr_c = left_matrix_dimensions[0];
    context.dot_product_length = left_matrix_dimensions[1];
    context.nc_c = right_matrix_dimensions[1];
    if (context.dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_avx_interleave_matrix_multiply:d", "Inconsistent matrix dimensions");
    }

    if ((context.nr_c <= NR_C_MAX) && (context.dot_product_length == 8))
    {
        context.left_matrix = copy_mx_to_cf32_matrix (left_matrix_in, &context.left_matrix_tcols);
        context.right_matrix = copy_mx_to_cf32_matrix (right_matrix_in, &context.right_matrix_tcols);
        context.output_matrix_tcols = (context.nc_c + 3) & ~3;
        context.output_matrix = mxCalloc (context.nr_c * context.output_matrix_tcols, sizeof(SAL_cf32));

        timing_results = time_matrix_multiply (timed_c_matrix_multiply, &context, mxGetScalar (num_timed_iterations_in));
        if (context.rc != SAL_SUCCESS)
        {
            mexErrMsgIdAndTxt ("c_avx_interleave_matrix_multiply:e", "cmat_mulx failed with rc=%u", context.rc);
        }

        mx_output_matrix = mxCreateNumericMatrix (context.nr_c, context.nc_c, mxSINGLE_CLASS, mxCOMPLEX);
        copy_cf32_to_mx_matrix (context.output_matrix, context.output_matrix_tcols, mx_output_matrix);
    
        mxFree (context.left_matrix);
        mxFree (context.right_matrix);
        mxFree (context.output_matrix);
    }
    else
    {
        /* Empty output to indicate the maxtrix dimension isn't support */
        mx_output_matrix = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxCOMPLEX);
        timing_results = mxCreateNumericMatrix (0, 0, mxUINT64_CLASS, mxREAL);
    }
    plhs[0] = mx_output_matrix;
    plhs[1] = timing_results;
}