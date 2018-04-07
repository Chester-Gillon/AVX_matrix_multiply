
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

typedef struct
{
    matrix_storage left_matrix, right_matrix, output_matrix;
    SAL_zf32 *left_matrix_rows;
    SAL_zf32 *right_matrix_rows;
    SAL_zf32 *output_matrix_rows;
    SAL_i32 nr_c, nc_c, dot_product_length;
} matrix_context;

static void timed_c_matrix_multiply (void *arg)
{
    matrix_context *const context = (matrix_context *) arg;
    mwSize r_c, c_c, dot_product;
    
    for (r_c = 0; r_c < context->nr_c; r_c++)
    {
        for (c_c = 0; c_c < context->nc_c; c_c++)
        {
            context->output_matrix_rows[r_c].realp[c_c] = 0.0f;
            context->output_matrix_rows[r_c].imagp[c_c] = 0.0f;
            for (dot_product = 0; dot_product < context->dot_product_length; dot_product++)
            {
                context->output_matrix_rows[r_c].realp[c_c] +=
                        (context->right_matrix_rows[dot_product].realp[c_c] * context->left_matrix_rows[r_c].realp[dot_product]) -
                        (context->right_matrix_rows[dot_product].imagp[c_c] * context->left_matrix_rows[r_c].imagp[dot_product]);
                context->output_matrix_rows[r_c].imagp[c_c] +=
                        (context->right_matrix_rows[dot_product].imagp[c_c] * context->left_matrix_rows[r_c].realp[dot_product]) +
                        (context->right_matrix_rows[dot_product].realp[c_c] * context->left_matrix_rows[r_c].imagp[dot_product]);
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const left_matrix_in = prhs[0];
    const mxArray *const right_matrix_in = prhs[1];
    const mxArray *const num_timed_iterations_in = prhs[2];
    const mxArray *const block_other_cpus_in = prhs[3];
    const mxArray *const alloc_method_in = prhs[4];
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    mxArray *mx_output_matrix;
    mxArray *timing_results;
    matrix_context context;

    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 5)
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (left_matrix_in) || !mxIsSingle (left_matrix_in) ||
        !mxIsComplex (right_matrix_in) || !mxIsSingle (right_matrix_in) ||
        (mxGetNumberOfDimensions (left_matrix_in) != 2) ||
        (mxGetNumberOfDimensions (right_matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    set_matrix_allocation_method (alloc_method_in);
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    context.nr_c = left_matrix_dimensions[0];
    context.dot_product_length = left_matrix_dimensions[1];
    context.nc_c = right_matrix_dimensions[1];
    if (context.dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:d", "Inconsistent matrix dimensions");
    }
    
    context.left_matrix_rows = mxCalloc (context.nr_c, sizeof (SAL_zf32));
    context.right_matrix_rows = mxCalloc (context.dot_product_length, sizeof (SAL_zf32));
    context.output_matrix_rows = mxCalloc (context.nr_c, sizeof (SAL_zf32));
    copy_mx_to_zf32_matrix (left_matrix_in, &context.left_matrix, context.left_matrix_rows);
    copy_mx_to_zf32_matrix (right_matrix_in, &context.right_matrix, context.right_matrix_rows);
    allocate_zf32_matrix (context.nr_c, context.nc_c, &context.output_matrix, context.output_matrix_rows);
    
    timing_results = time_matrix_multiply (timed_c_matrix_multiply, &context, mxGetScalar (num_timed_iterations_in),
                                           mxGetScalar (block_other_cpus_in));

    mx_output_matrix = copy_zf32_to_mx_matrix (&context.output_matrix);
    plhs[0] = mx_output_matrix;
    plhs[1] = timing_results;
    
    free_matrix (&context.left_matrix);
    free_matrix (&context.right_matrix);
    free_matrix (&context.output_matrix);
    mxFree (context.left_matrix_rows);
    mxFree (context.right_matrix_rows);
    mxFree (context.output_matrix_rows);
}