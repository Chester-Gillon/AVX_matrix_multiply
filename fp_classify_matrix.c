/* Return a structure which counts the classification of the 32-bit values in a complex 2D matrix */

#include <math.h>

#include <sal.h>

#include <mex.h>

static mxArray *create_ui32_scalar (const SAL_ui32 value)
{
    mxArray *const array = mxCreateNumericMatrix (1, 1, mxUINT32_CLASS, mxREAL);
    long *const array_data = mxGetData (array);
    
    *array_data = value;
    return array;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const matrix_in = prhs[0];
    SAL_ui32 count_nan = 0;
    SAL_ui32 count_infinite = 0;
    SAL_ui32 count_zero = 0;
    SAL_ui32 count_subnormal = 0;
    SAL_ui32 count_normal = 0;
    mwSize cell_index;
    int real_or_imag;
    const char *field_names[] = {"count_nan", "count_infinite", "count_zero", "count_subnormal", "count_normal"};
    mxArray *const classification_struct = mxCreateStructMatrix (1, 1, 5, field_names);

    if (nlhs != 1)
    {
        mexErrMsgIdAndTxt ("fp_classify_matrix:a", "Incorrect number of outputs");
    }
    if (nrhs != 1)
    {
        mexErrMsgIdAndTxt ("fp_classify_matrix:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (matrix_in) || !mxIsSingle (matrix_in) ||
        (mxGetNumberOfDimensions (matrix_in) != 2))
    {
        mexErrMsgIdAndTxt ("fp_classify_matrix:c", "Input is not a complex single 2D array");
    }

    const mwSize *const matrix_dimensions = mxGetDimensions (matrix_in);
    const mwSize num_cells = matrix_dimensions[0] * matrix_dimensions[1];
    const float *const data[2] = {mxGetData (matrix_in), mxGetImagData (matrix_in)};
    
    for (real_or_imag = 0; real_or_imag < 2; real_or_imag++)
    {
        for (cell_index = 0; cell_index < num_cells; cell_index++)
        {
            switch (fpclassify (data[real_or_imag][cell_index]))
            {
            case FP_NAN:
                count_nan++;
                break;
                
            case FP_INFINITE:
                count_infinite++;
                break;
                
            case FP_ZERO:
                count_zero++;
                break;
                
            case FP_SUBNORMAL: 
                count_subnormal++;
                break;
                
            case FP_NORMAL:
                count_normal++;
                break;
            }
        }
    }

    mxSetField (classification_struct, 0, "count_nan", create_ui32_scalar (count_nan));
    mxSetField (classification_struct, 0, "count_infinite", create_ui32_scalar (count_infinite));
    mxSetField (classification_struct, 0, "count_zero", create_ui32_scalar (count_zero));
    mxSetField (classification_struct, 0, "count_subnormal", create_ui32_scalar (count_subnormal));
    mxSetField (classification_struct, 0, "count_normal", create_ui32_scalar (count_normal));
    plhs[0] = classification_struct;
}
