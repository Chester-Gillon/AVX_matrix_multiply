#include <pthread.h>
#include <time.h>

#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

typedef struct
{
    void (*matrix_func) (void *);
    SAL_i32 num_timed_iterations;
    SAL_i64 min_duration_us;
    SAL_i64 max_duration_us;
    SAL_i64 total_duration_us;
    void *test_specific_context;
} timed_thread_data;

SAL_cf32* copy_mx_to_cf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols)
{
    const float *real_in = mxGetData (mx_matrix);
    const float *imag_in = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    SAL_cf32 *C_matrix;
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    /* Align each C matrix column to start on an AVX aligned boundrary */
    *tcols = (num_cols + 3) & ~3;
    C_matrix = mxCalloc (num_rows * *tcols, sizeof(SAL_cf32));
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * *tcols) + col;
            col_major_index = (col * num_rows) + row;
            C_matrix[row_major_index].real = real_in[col_major_index];
            C_matrix[row_major_index].imag = imag_in[col_major_index];
        }
    }
    
    return C_matrix;
}

void copy_cf32_to_mx_matrix (const SAL_cf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix)
{
    float *real_out = mxGetData (mx_matrix);
    float *imag_out = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * tcols) + col;
            col_major_index = (col * num_rows) + row;
            real_out[col_major_index] = C_matrix[row_major_index].real;
            imag_out[col_major_index] = C_matrix[row_major_index].imag;
        }
    }
}

void copy_mx_to_zf32_matrix (const mxArray *const mx_matrix, SAL_i32 *const tcols, SAL_zf32 *const C_matrix)
{
    const float *real_in = mxGetData (mx_matrix);
    const float *imag_in = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    /* Align each C matrix column to start on an AVX aligned boundrary */
    *tcols = (num_cols + 7) & ~7;
    C_matrix->realp = mxCalloc (num_rows * *tcols, sizeof(float));
    C_matrix->imagp = mxCalloc (num_rows * *tcols, sizeof(float));
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * *tcols) + col;
            col_major_index = (col * num_rows) + row;
            C_matrix->realp[row_major_index] = real_in[col_major_index];
            C_matrix->imagp[row_major_index] = imag_in[col_major_index];
        }
    }
}

void copy_zf32_to_mx_matrix (const SAL_zf32 *const C_matrix, const SAL_i32 tcols,
                             mxArray *const mx_matrix)
{
    float *real_out = mxGetData (mx_matrix);
    float *imag_out = mxGetImagData (mx_matrix);
    const mwSize *const matrix_dimensions = mxGetDimensions (mx_matrix);
    const mwSize num_rows = matrix_dimensions[0];
    const mwSize num_cols = matrix_dimensions[1];
    mwSize row, col;
    mwSize row_major_index, col_major_index;
    
    for (row = 0; row < num_rows; row++)
    {
        for (col = 0; col < num_cols; col++)
        {
            row_major_index = (row * tcols) + col;
            col_major_index = (col * num_rows) + row;
            real_out[col_major_index] = C_matrix->realp[row_major_index];
            imag_out[col_major_index] = C_matrix->imagp[row_major_index];
        }
    }
}

static void *timing_thread (void * arg)
{
    timed_thread_data *const thread_data = (timed_thread_data *) arg;
    SAL_i32 warmup_iter;
    SAL_i32 timed_iter;
    SAL_i64 duration_us;
    struct timespec start_time, stop_time;
    
    /* Perform a warm-up run to try and get code / data cached for a more
     * representative comparision between different matrix multiply implementations.
     * e.g. since we haven't attempted to use mlockall */
    for (warmup_iter = 0; warmup_iter < 2; warmup_iter++)
    {
        clock_gettime (CLOCK_MONOTONIC_RAW, &start_time);
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        clock_gettime (CLOCK_MONOTONIC_RAW, &stop_time);
    }
    
    for (timed_iter = 0; timed_iter < thread_data->num_timed_iterations; timed_iter++)
    {
        clock_gettime (CLOCK_MONOTONIC_RAW, &start_time);
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        clock_gettime (CLOCK_MONOTONIC_RAW, &stop_time);
        duration_us = ((stop_time.tv_sec  * 1000000) + (stop_time.tv_nsec  / 1000)) -
                      ((start_time.tv_sec * 1000000) + (start_time.tv_nsec / 1000));
        if (timed_iter == 0)
        {
            thread_data->min_duration_us = duration_us;
            thread_data->max_duration_us = duration_us;
        }
        else
        {
            if (duration_us < thread_data->min_duration_us)
            {
                thread_data->min_duration_us = duration_us;
            }
            else if (duration_us > thread_data->max_duration_us)
            {
                thread_data->max_duration_us = duration_us;
            }
        }
        thread_data->total_duration_us += duration_us;
    }
    
    return NULL;
}

mxArray *time_matrix_multiply (void (*matrix_func) (void *),
                               void *test_specific_context,
                               SAL_i32 num_timed_iterations)
{
    timed_thread_data thread_data;
    pthread_t thread_id;
    pthread_attr_t attr;
    void *ret_val;
    int rc;
    const char *field_names[] = {"min_duration_us", "max_duration_us", "average_duration_us"};
    mxArray *timing_results;
    
    thread_data.matrix_func = matrix_func;
    thread_data.test_specific_context = test_specific_context;
    thread_data.num_timed_iterations = num_timed_iterations;
    thread_data.min_duration_us = 0;
    thread_data.max_duration_us = 0;
    thread_data.total_duration_us = 0;
    
    rc = pthread_attr_init (&attr);
    mxAssertS (rc == 0, "pthread_attr_init");
    
    rc = pthread_create (&thread_id, &attr, timing_thread, &thread_data);
    mxAssertS (rc == 0, "pthread_create");
    
    rc = pthread_join (thread_id, &ret_val);
    mxAssertS (rc == 0, "pthread_join");
    
    rc = pthread_attr_destroy (&attr);
    mxAssertS (rc == 0, "pthread_attr_destroy");
    
    timing_results = mxCreateStructMatrix (1, 1, 3, field_names);
    mxSetFieldByNumber (timing_results, 0, 0, mxCreateDoubleScalar (thread_data.min_duration_us));
    mxSetFieldByNumber (timing_results, 0, 1, mxCreateDoubleScalar (thread_data.max_duration_us));
    mxSetFieldByNumber (timing_results, 0, 2,
            mxCreateDoubleScalar ((double) thread_data.total_duration_us / (double) num_timed_iterations));
    
    return timing_results;
}
