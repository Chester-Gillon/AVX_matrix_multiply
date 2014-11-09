#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <semaphore.h>
#include <sys/mman.h>

#include <mex.h>
#include <matrix.h>

#include <sal.h>

#include "matrix_utils.h"

#define MAX_CPUS 32

typedef struct
{
    sem_t ready_sem;
    sem_t go_sem;
} blocking_thread_sems;

typedef struct
{
    struct timespec start_time;
    struct timespec stop_time;
} test_times;

typedef struct
{
    void (*matrix_func) (void *);
    SAL_i32 num_timed_iterations;
    test_times *times;
    void *test_specific_context;
    SAL_i32 num_blocked_cpus;
    blocking_thread_sems blocking_sems[MAX_CPUS];
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

static void *cpu_blocking_thread (void *arg)
{
    blocking_thread_sems *const blocking_sems = (blocking_thread_sems *) arg;
    int rc;
    
    rc = sem_post (&blocking_sems->ready_sem);
    mxAssertS (rc == 0, "sem_post");
    
    rc = sem_wait (&blocking_sems->go_sem);
    mxAssertS (rc == 0, "sem_wait");
    
    do
    {
        rc = sem_trywait (&blocking_sems->go_sem);
    } while (rc != 0);
    
    return NULL;
}

static void *timing_thread (void *arg)
{
    timed_thread_data *const thread_data = (timed_thread_data *) arg;
    SAL_i32 warmup_iter;
    SAL_i32 timed_iter;
    struct timespec start_time, stop_time;
    SAL_i32 blocking_thread_index;
    int rc;
    
    /* Request the blocking threads running on the other CPUs to spin in a
     * tight loop at the same real-time priority as this thread during the
     * duration of the timing test.
     * This is try and prevent other CPUs from consuming L3 / memory bandwidth
     * during the duration of the timing test. */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        rc = sem_wait (&thread_data->blocking_sems[blocking_thread_index].ready_sem);
        mxAssertS (rc == 0, "sem_wait");
    }

    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        rc = sem_post (&thread_data->blocking_sems[blocking_thread_index].go_sem);
        mxAssertS (rc == 0, "sem_post");
    }
    
    /* Perform a warm-up run to try and get code / data cached for a more
     * representative comparision between different matrix multiply implementations. */
    for (warmup_iter = 0; warmup_iter < 2; warmup_iter++)
    {
        clock_gettime (CLOCK_MONOTONIC_RAW, &start_time);
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        clock_gettime (CLOCK_MONOTONIC_RAW, &stop_time);
    }
    
    for (timed_iter = 0; timed_iter < thread_data->num_timed_iterations; timed_iter++)
    {
        clock_gettime (CLOCK_MONOTONIC_RAW, &thread_data->times[timed_iter].start_time);
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        clock_gettime (CLOCK_MONOTONIC_RAW, &thread_data->times[timed_iter].stop_time);
    }
    
    /* Tell the blocking threads to exit */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        rc = sem_post (&thread_data->blocking_sems[blocking_thread_index].go_sem);
        mxAssertS (rc == 0, "sem_post");
    }
    
    return NULL;
}

mxArray *time_matrix_multiply (void (*matrix_func) (void *),
                               void *test_specific_context,
                               const SAL_i32 num_timed_iterations,
                               const bool block_other_cpus)
{
    timed_thread_data thread_data;
    pthread_t timing_thread_id;
    pthread_t blocking_thread_ids[MAX_CPUS];
    pthread_attr_t attr;
    void *ret_val;
    int rc;
    mxArray *timing_results;
    cpu_set_t cpu_set;
    struct sched_param param;
    const SAL_i32 num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    SAL_i32 blocking_thread_index;
    SAL_i32 timed_iter;
    mxArray *start_times_tv_sec;
    SAL_i32 *start_times_tv_sec_data;
    mxArray *start_times_tv_nsec;
    SAL_i32 *start_times_tv_nsec_data;
    mxArray *stop_times_tv_sec;
    SAL_i32 *stop_times_tv_sec_data;
    mxArray *stop_times_tv_nsec;
    SAL_i32 *stop_times_tv_nsec_data;
    const char *field_names[] = {"start_times_tv_sec", "start_times_tv_nsec", "stop_times_tv_sec", "stop_times_tv_nsec"};
    
    rc = mlockall (MCL_CURRENT | MCL_FUTURE);
    mxAssertS (rc == 0, "mlockall");
    
    thread_data.matrix_func = matrix_func;
    thread_data.test_specific_context = test_specific_context;
    thread_data.num_timed_iterations = num_timed_iterations;
    thread_data.times = mxCalloc (num_timed_iterations, sizeof (test_times));
    thread_data.num_blocked_cpus = block_other_cpus ? num_cpus - 1 : 0;
    
    for (blocking_thread_index = 0; blocking_thread_index < thread_data.num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data.blocking_sems[blocking_thread_index];
        rc = sem_init (&blocking_sems->ready_sem, 0, 0);
        mxAssertS (rc == 0, "sem_init");
        rc = sem_init (&blocking_sems->go_sem, 0, 0);
        mxAssertS (rc == 0, "sem_init");
    }
    
    rc = pthread_attr_init (&attr);
    mxAssertS (rc == 0, "pthread_attr_init");
    
    /* Create threads as real-time to minimise interruptions */
    rc = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
    mxAssertS (rc == 0, "pthread_attr_setinheritsched");
    
    rc = pthread_attr_setschedpolicy (&attr, SCHED_FIFO);
    mxAssertS (rc == 0, "pthread_attr_setschedpolicy");
    
    param.sched_priority = 49;
    rc = pthread_attr_setschedparam (&attr, &param);
    mxAssertS (rc == 0, "pthread_attr_setschedparam");
    
    /* Run thread only on highest numbered CPU - which can be isolated from
     * being used for other processes using tuna */
    mxAssertS (num_cpus <= MAX_CPUS, "sysconf(_SC_NPROCESSORS_ONLN) larger than MAX_CPUS");
    CPU_ZERO (&cpu_set);
    CPU_SET (num_cpus - 1, &cpu_set);
    rc = pthread_attr_setaffinity_np (&attr, sizeof(cpu_set), &cpu_set);
    mxAssertS (rc == 0, "pthread_attr_setaffinity_np");
    
    rc = pthread_create (&timing_thread_id, &attr, timing_thread, &thread_data);
    if (rc != 0)
    {
        mexErrMsgIdAndTxt ("time_matrix_multiply:pthread_create",
                "This can fail due to insufficient permission to set rtprio : rc=%d, error=%s", rc, sys_errlist[rc]);
    }
    
    /* Start blocking threads on all CPUs not used for the timing test */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data.num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data.blocking_sems[blocking_thread_index];

        CPU_ZERO (&cpu_set);
        CPU_SET (blocking_thread_index, &cpu_set);
        rc = pthread_attr_setaffinity_np (&attr, sizeof(cpu_set), &cpu_set);
        mxAssertS (rc == 0, "pthread_attr_setaffinity_np");
            
        rc = pthread_create (&blocking_thread_ids[blocking_thread_index], &attr, cpu_blocking_thread, blocking_sems);
        mxAssertS (rc == 0, "pthread_create");
    }
    
    rc = pthread_join (timing_thread_id, &ret_val);
    mxAssertS (rc == 0, "pthread_join");
    
    /* Terminate blocking threads */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data.num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data.blocking_sems[blocking_thread_index];

        rc = pthread_join (blocking_thread_ids[blocking_thread_index], &ret_val);
        mxAssertS (rc == 0, "pthread_join");
            
        rc = sem_destroy (&blocking_sems->ready_sem);
        mxAssertS (rc == 0, "sem_destroy");
        rc = sem_destroy (&blocking_sems->go_sem);
        mxAssertS (rc == 0, "sem_destroy");
    }
    
    rc = pthread_attr_destroy (&attr);
    mxAssertS (rc == 0, "pthread_attr_destroy");

    start_times_tv_sec = mxCreateNumericMatrix (1, num_timed_iterations, mxINT32_CLASS, mxREAL);
    start_times_tv_nsec = mxCreateNumericMatrix (1, num_timed_iterations, mxINT32_CLASS, mxREAL);
    stop_times_tv_sec = mxCreateNumericMatrix (1, num_timed_iterations, mxINT32_CLASS, mxREAL);
    stop_times_tv_nsec = mxCreateNumericMatrix (1, num_timed_iterations, mxINT32_CLASS, mxREAL);
    start_times_tv_sec_data = mxGetData (start_times_tv_sec);
    start_times_tv_nsec_data = mxGetData (start_times_tv_nsec);
    stop_times_tv_sec_data = mxGetData (stop_times_tv_sec);
    stop_times_tv_nsec_data = mxGetData (stop_times_tv_nsec);
    for (timed_iter = 0; timed_iter < num_timed_iterations; timed_iter++)
    {
        start_times_tv_sec_data[timed_iter] = thread_data.times[timed_iter].start_time.tv_sec;
        start_times_tv_nsec_data[timed_iter] = thread_data.times[timed_iter].start_time.tv_nsec;
        stop_times_tv_sec_data[timed_iter] = thread_data.times[timed_iter].stop_time.tv_sec;
        stop_times_tv_nsec_data[timed_iter] = thread_data.times[timed_iter].stop_time.tv_nsec;
    }
    mxFree (thread_data.times);
    
    timing_results = mxCreateStructMatrix (1, 1, 4, field_names);
    mxSetField (timing_results, 0, "start_times_tv_sec", start_times_tv_sec);
    mxSetField (timing_results, 0, "start_times_tv_nsec", start_times_tv_nsec);
    mxSetField (timing_results, 0, "stop_times_tv_sec", stop_times_tv_sec);
    mxSetField (timing_results, 0, "stop_times_tv_nsec", stop_times_tv_nsec);

    return timing_results;
}
