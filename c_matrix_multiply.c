
#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

typedef struct
{
    const float *left_matrix_real, *left_matrix_imag;
    const float *right_matrix_real, *right_matrix_imag;
    float *output_matrix_real, *output_matrix_imag;
    SAL_i32 nr_c, nc_c, dot_product_length;
} matrix_context;

static void timed_c_matrix_multiply (void *arg)
{
    matrix_context *const context = (matrix_context *) arg;
    mwSize left_matrix_index, right_matrix_index, output_matrix_index;
    mwSize r_c, c_c, dot_product;
    
    for (r_c = 0; r_c < context->nr_c; r_c++)
    {
        for (c_c = 0; c_c < context->nc_c; c_c++)
        {
            output_matrix_index = r_c + (c_c * context->nr_c);
            context->output_matrix_real[output_matrix_index] = 0.0f;
            context->output_matrix_imag[output_matrix_index] = 0.0f;
            for (dot_product = 0; dot_product < context->dot_product_length; dot_product++)
            {
                left_matrix_index = r_c + (dot_product * context->nr_c);
                right_matrix_index = dot_product + (c_c * context->dot_product_length);
                context->output_matrix_real[output_matrix_index] +=
                        (context->right_matrix_real[right_matrix_index] * context->left_matrix_real[left_matrix_index]) -
                        (context->right_matrix_imag[right_matrix_index] * context->left_matrix_imag[left_matrix_index]);
                context->output_matrix_imag[output_matrix_index] +=
                        (context->right_matrix_imag[right_matrix_index] * context->left_matrix_real[left_matrix_index]) +
                        (context->right_matrix_real[right_matrix_index] * context->left_matrix_imag[left_matrix_index]);
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
    const mwSize *left_matrix_dimensions;
    const mwSize *right_matrix_dimensions;
    mxArray *output_matrix;
    matrix_context context;

    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 4)
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
    
    left_matrix_dimensions = mxGetDimensions (left_matrix_in);
    right_matrix_dimensions = mxGetDimensions (right_matrix_in);
    context.nr_c = left_matrix_dimensions[0];
    context.dot_product_length = left_matrix_dimensions[1];
    context.nc_c = right_matrix_dimensions[1];
    if (context.dot_product_length != right_matrix_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:d", "Inconsistent matrix dimensions");
    }
    
    output_matrix = mxCreateNumericMatrix (context.nr_c, context.nc_c, mxSINGLE_CLASS, mxCOMPLEX);
    plhs[0] = output_matrix;
    
    context.left_matrix_real = mxGetData (left_matrix_in);
    context.left_matrix_imag = mxGetImagData (left_matrix_in);
    context.right_matrix_real = mxGetData (right_matrix_in);
    context.right_matrix_imag = mxGetImagData (right_matrix_in);
    context.output_matrix_real = mxGetData (output_matrix);
    context.output_matrix_imag = mxGetImagData (output_matrix);
    
    plhs[1] = time_matrix_multiply (timed_c_matrix_multiply, &context, mxGetScalar (num_timed_iterations_in),
                                    mxGetScalar (block_other_cpus_in));
}