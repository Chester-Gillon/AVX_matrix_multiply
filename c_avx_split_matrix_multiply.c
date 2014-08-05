
#include <mex.h>
#include <matrix.h>

#include "avx_matrix_multiply_library.h"

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
    packed_8_split *weights;
    split *samples;
    split *outputs;
 
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
    
    weights = mxCalloc (num_sets, sizeof (packed_8_split));
    weights_real = mxGetData (weights_in);
    weights_imag = mxGetImagData (weights_in);
    for (set = 0; set < num_sets; set++)
    {
        weights[set].real = mxCalloc (num_sets, sizeof (packed_8));
        weights[set].imag = mxCalloc (num_sets, sizeof (packed_8));
        for (weight = 0; weight < num_weights; weight++)
        {
            weight_index = set + (weight * num_sets);
            weights[set].real->data[weight] = weights_real[weight_index];
            weights[set].imag->data[weight] = weights_imag[weight_index];
        }
    }

    samples = mxCalloc (num_weights, sizeof (split));
    samples_real = mxGetData (samples_in);
    samples_imag = mxGetImagData (samples_in);
    for (weight = 0; weight < num_weights; weight++)
    {
        samples[weight].real = mxCalloc (num_samples + num_weights, sizeof(float));
        samples[weight].imag = mxCalloc (num_samples + num_weights, sizeof(float));
        for (sample = 0; sample < num_samples; sample++)
        {
            sample_index = weight + (sample * num_weights);
            samples[weight].real[sample] = samples_real[sample_index];
            samples[weight].imag[sample] = samples_imag[sample_index];
        }
    }
    
    outputs = mxCalloc (num_sets, sizeof (split));
    for (set = 0; set < num_sets; set++)
    {
        outputs[set].real = mxCalloc (num_samples + num_weights, sizeof(float));
        outputs[set].imag = mxCalloc (num_samples + num_weights, sizeof(float));
    }

    packed_8_split_matrix_multiply (weights, samples, outputs, num_samples, num_sets);

    output_matrix = mxCreateNumericMatrix (num_sets, num_samples, mxSINGLE_CLASS, mxCOMPLEX);
    plhs[0] = output_matrix;

    output_real = mxGetData (output_matrix);
    output_imag = mxGetImagData (output_matrix);
    for (set = 0; set < num_sets; set++)
    {
        for (sample = 0; sample < num_samples; sample++)
        {
            output_index = set + (sample * num_sets);
            output_real[output_index] = outputs[set].real[sample];
            output_imag[output_index] = outputs[set].imag[sample];
        }
    }

    for (set = 0; set < num_sets; set++)
    {
        mxFree (outputs[set].real);
        mxFree (outputs[set].imag);
    }
    mxFree (outputs);

    for (weight = 0; weight < num_weights; weight++)
    {
        mxFree (samples[weight].real);
        mxFree (samples[weight].imag);
    }
    mxFree (samples);

    for (set = 0; set < num_sets; set++)
    {
        mxFree (weights[set].real);
        mxFree (weights[set].imag);
    }
    mxFree (weights);
}