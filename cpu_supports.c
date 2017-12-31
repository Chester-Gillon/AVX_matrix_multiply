/* Return a boolean which indicates if a given CPU feature is supported.
 * Where the feature names are defined by the GCC __builtin_cpu_supports()
 **/

#include <string.h>

#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *feature_name;
    int feature_result;
    
    if ((nrhs != 1) || !mxIsChar (prhs[0]) || (mxGetNumberOfDimensions (prhs[0]) != 2))
    {
        mexErrMsgIdAndTxt ("cpu_supports:a", "Input must be a character array");        
    }
    if (nlhs != 1)
    {
        mexErrMsgIdAndTxt ("cpu_supports:b", "Must have a single output");        
    }
    
    feature_name = mxArrayToString (prhs[0]);
    __builtin_cpu_init ();
    if (strcmp (feature_name, "avx") == 0)
    {
        feature_result = __builtin_cpu_supports ("avx");
    }
    else if (strcmp (feature_name, "avx2") == 0)
    {
        feature_result = __builtin_cpu_supports ("avx2");
    }
    else if (strcmp (feature_name, "fma") == 0)
    {
        feature_result = __builtin_cpu_supports ("fma");
    }
    else
    {
        mexErrMsgIdAndTxt ("cpu_supports:c", "Unknown CPU feature");
        feature_result = 0;
    }
    plhs[0] = mxCreateLogicalScalar (feature_result != 0);
    mxFree (feature_name);
}
