
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "cmat_mulx_fixed_dimension_fma_accumulate_matrix_multiplies.h"
#include "matrix_utils.h"

typedef struct
{
    SAL_i32 nr_c, nc_c, dot_product_length;
    SAL_cf32 *left_matrix_rows[CMAT_MULX_FIXED_DIMENSION_FMA_ACCUMULATE_MAX_NR_C];
    SAL_cf32 *right_matrix_rows[CMAT_MULX_FIXED_DIMENSION_FMA_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH];
    SAL_cf32 *output_matrix_rows[CMAT_MULX_FIXED_DIMENSION_FMA_ACCUMULATE_MAX_NR_C];
    cmat_mulx_fixed_dimension_fma_accumulate_func cmat_mulx_func;
} matrix_context;

static void timed_c_matrix_multiply (void *arg)
{
    matrix_context *const context = (matrix_context *) arg;

    (*context->cmat_mulx_func) (context->left_matrix_rows,
                                context->right_matrix_rows,
                                context->output_matrix_rows,
                                context->nc_c);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const left_matrix_in = prhs[0];
    const mxArray *const right_matrix_in = prhs[1];
    const mxArray *const num_timed_iterations_in = prhs[2];
    const mxArray *const block_other_cpus_in = prhs[3];
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    matrix_storage left_matrix, right_matrix, output_matrix;
    mxArray *mx_output_matrix;
    mxArray *timing_results;
    matrix_context context;

    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt ("c_avx_fixed_dimension_fma_accumulate_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 4)
    {
        mexErrMsgIdAndTxt ("c_avx_fixed_dimension_fma_accumulate_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (left_matrix_in) || !mxIsSingle (left_matrix_in) ||
        !mxIsComplex (right_matrix_in) || !mxIsSingle (right_matrix_in) ||
        (mxGetNumberOfDimensions (left_matrix_in) != 2) ||
        (mxGetNumberOfDimensions (right_matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_avx_fixed_dimension_fma_accumulate_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    context.nr_c = left_matrix_dimensions[0];
    context.dot_product_length = left_matrix_dimensions[1];
    context.nc_c = right_matrix_dimensions[1];
    if (context.dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_avx_fixed_dimension_fma_accumulate_matrix_multiply:d", "Inconsistent matrix dimensions");
    }

    context.cmat_mulx_func = NULL;
    if ((context.nr_c <= CMAT_MULX_FIXED_DIMENSION_FMA_ACCUMULATE_MAX_NR_C) &&
        (context.dot_product_length <= CMAT_MULX_FIXED_DIMENSION_FMA_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH))
    {
        context.cmat_mulx_func = cmat_mulx_fixed_dimension_fma_accumulate_functions[context.nr_c][context.dot_product_length];
    }
    
    if (context.cmat_mulx_func != NULL)
    {
        copy_mx_to_cf32_matrix (left_matrix_in, &left_matrix, context.left_matrix_rows);
        copy_mx_to_cf32_matrix (right_matrix_in, &right_matrix, context.right_matrix_rows);
        allocate_cf32_matrix (context.nr_c, context.nc_c, &output_matrix, context.output_matrix_rows);

        timing_results = time_matrix_multiply (timed_c_matrix_multiply, &context, mxGetScalar (num_timed_iterations_in),
                                               mxGetScalar (block_other_cpus_in));

        mx_output_matrix = copy_cf32_to_mx_matrix (&output_matrix);
    
        free_matrix (&left_matrix);
        free_matrix (&right_matrix);
        free_matrix (&output_matrix);
    }
    else
    {
        /* Empty output matrix indicates fixed dimensions not support */
        mx_output_matrix = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxCOMPLEX);
        timing_results = mxCreateStructMatrix (0, 0, 0, NULL);
    }
    plhs[0] = mx_output_matrix;
    plhs[1] = timing_results;
}