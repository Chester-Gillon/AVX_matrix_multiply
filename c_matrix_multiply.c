
#include <mex.h>
#include <matrix.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *const weights_in = prhs[0];
    const mxArray *const samples_in = prhs[1];
    const mwSize *weights_dimensions;
    const mwSize *samples_dimensions;
    mwSize num_samples, num_weights, num_sets;
    mwSize set, sample, weight;
    mwSize weight_index, sample_index, output_index;
    mxArray *output_matrix;
    const float *weights_real, *weights_imag;
    const float *samples_real, *samples_imag;
    float *output_real, *output_imag;

    if (nlhs != 1)
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:a", "Incorrect number of outputs");
    }
    if (nrhs != 2)
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:b", "Incorrect number of inputs");
    }
    
    if (!mxIsComplex (weights_in) || !mxIsSingle (weights_in) ||
        !mxIsComplex (samples_in) || !mxIsSingle (samples_in) ||
        (mxGetNumberOfDimensions (weights_in) != 2) ||
        (mxGetNumberOfDimensions (samples_in) != 2))
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:c", "Inputs are not complex single 2D arrays");
    }
    
    weights_dimensions = mxGetDimensions (weights_in);
    samples_dimensions = mxGetDimensions (samples_in);
    num_sets = weights_dimensions[0];
    num_weights = weights_dimensions[1];
    num_samples = samples_dimensions[1];
    if (num_weights != samples_dimensions[0])
    {
        mexErrMsgIdAndTxt ("c_matrix_multiply:d", "Inconsistent number of weights");
    }
    
    output_matrix = mxCreateNumericMatrix (num_sets, num_samples, mxSINGLE_CLASS, mxCOMPLEX);
    plhs[0] = output_matrix;
    
    weights_real = mxGetData (weights_in);
    weights_imag = mxGetImagData (weights_in);
    samples_real = mxGetData (samples_in);
    samples_imag = mxGetImagData (samples_in);
    output_real = mxGetData (output_matrix);
    output_imag = mxGetImagData (output_matrix);
    
    for (set = 0; set < num_sets; set++)
    {
        for (sample = 0; sample < num_samples; sample++)
        {
            output_index = set + (sample * num_sets);
            output_real[output_index] = 0.0f;
            output_imag[output_index] = 0.0f;
            for (weight = 0; weight < num_weights; weight++)
            {
                weight_index = set + (weight * num_sets);
                sample_index = weight + (sample * num_weights);
                output_real[output_index] +=
                        (samples_real[sample_index] * weights_real[weight_index]) -
                        (samples_imag[sample_index] * weights_imag[weight_index]);
                output_imag[output_index] +=
                        (samples_imag[sample_index] * weights_real[weight_index]) +
                        (samples_real[sample_index] * weights_imag[weight_index]);
            }
        }
    }
}