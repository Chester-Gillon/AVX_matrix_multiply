/* The MATLAB documentation doesn't state what memory alignment the mxMalloc and mxCalloc functions provide.
 * Since AVX vectors need 32-byte alignment this mex file was created to perform allocations of increasing
 * sizes and bit-wise OR the allocated pointers to deduce the alignment.
 *
 * MATLAB R2014A under Linux 64-bit was seen to provide 32-byte alignment.
 */

#include <mex.h>
#include <matrix.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int len;
    void *data;
    size_t data_addr;
    size_t alignment;
    
    alignment = 0;
    for (len = 1; len < 0x10001; len++)
    {
        data = mxMalloc (len);
        data_addr = (size_t) data;
        alignment |= data_addr;
    }
    
    mexPrintf ("mxMalloc alignment=%lx\n", alignment); 
    
    alignment = 0;
    for (len = 1; len < 0x10001; len++)
    {
        data = mxCalloc (len, 1);
        data_addr = (size_t) data;
        alignment |= data_addr;
    }
    
    mexPrintf ("mxCalloc alignment=%lx\n", alignment); 
}
