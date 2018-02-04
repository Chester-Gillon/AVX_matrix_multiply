#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <xmmintrin.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <asm/unistd.h>
#include <sched.h>

#include <mex.h>
#include <matrix.h>

#include <sal.h>

/* Has to be after sal.h to avoid conflicting definitions */
#include <sys/ioctl.h>

#include "matrix_utils.h"

#define MAX_CPUS 32

#define CACHE_LINE_SIZE 64

typedef struct
{
    sem_t ready_sem;
    sem_t go_sem;
    volatile bool exit_spin_loop;
} blocking_thread_sems __attribute__ ((aligned (CACHE_LINE_SIZE)));

typedef struct
{
    struct timespec start_time;
    struct timespec stop_time;
    SAL_ui64 start_inner_rdtsc;
    SAL_ui64 stop_inner_rdtsc;
    SAL_ui64 start_outer_rdtsc;
    SAL_ui64 stop_outer_rdtsc;
} test_times;

/* The hardware perf events captured as a group for the running process */
typedef enum
{
    /* Built in events */
    EVENT_HW_INSTRUCTIONS,
    EVENT_HW_REF_CPU_CYCLES,
    /* Select 4 events for L1D cache.
     * 4 is the maximum number of events which can be selected to count as a group when
     * hyper-threading is enabled.
     * If hyper-threading is disabled, the number of event counters doubles. */
    EVENT_L1D_READ_ACCESS,
    EVENT_L1D_READ_MISS,
    EVENT_L1D_WRITE_ACCESS,
    EVENT_L1D_WRITE_MISS,
    /* For sizing arrays */
    NUM_HW_PERF_EVENTS
} hw_perf_events;

/* One perf event read as a group */
typedef struct
{
    SAL_ui64 value; /* The value of the event */
    SAL_ui64 id;
} hw_event_group_event;

/* The structure of the perf events from a group read */
typedef struct
{
    SAL_ui64 num_events;
    SAL_ui64 time_enabled;
    SAL_ui64 time_running;
    hw_event_group_event events[NUM_HW_PERF_EVENTS];
} hw_events_group_format;

typedef struct
{
    void (*matrix_func) (void *);
    SAL_i32 num_timed_iterations;
    test_times *times;
    void *test_specific_context;
    SAL_i32 num_blocked_cpus;
    blocking_thread_sems blocking_sems[MAX_CPUS] __attribute__ ((aligned (CACHE_LINE_SIZE)));
    struct rusage start_self_usage;
    struct rusage start_thread_usage;
    struct rusage stop_self_usage;
    struct rusage stop_thread_usage;
    int perf_event_fds[NUM_HW_PERF_EVENTS];
    SAL_ui64 perf_event_values[NUM_HW_PERF_EVENTS];
} timed_thread_data;

/* Since there is no glibc wrapper for the system call */
long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                     int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);

    return ret;
}

static __inline__ SAL_ui64 read_rdtsc (void)
{
    unsigned long lo, hi;
    __asm__ __volatile__ ( "rdtsc" : "=a" (lo), "=d" (hi) );
    return( lo | (hi << 32) );
}

void *mxCalloc_and_touch (mwSize n, mwSize size)
{
    char *data = mxCalloc (n, size);
    
    if (data != NULL)
    {
        const mwSize total_bytes = n * size;
        const mwSize page_size = sysconf (_SC_PAGE_SIZE);
        mwSize index;
        
        for (index = 0; index < total_bytes; index += page_size)
        {
            data[index] = 0;
        }
        data[total_bytes-1] = 0;
    }
    
    return data;
}

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
        _mm_pause ();
    } while (!blocking_sems->exit_spin_loop);
    
    return NULL;
}

/* Open the perf events, as a group which are pinned to the CPU of the calling timing_thread,
 * and therefore should therefore not be multiplexed.
 *
 * @todo When initially opened the perf events from the main thread before the timing_thread
 *       then the perf events captured for the timing_thread were all zero, regardless
 *       of if perf_event_open() selected events for all CPUs or just the CPU used for
 *       the timing_thread. Moving the open of the perf events to the timing_thread
 *       allowed the events to be captured. */
static void open_perf_events (timed_thread_data *const thread_data)
{
    struct perf_event_attr event_attr;
    const int cpu = sched_getcpu ();

    /* Open the group leader event */
    memset (&event_attr, 0, sizeof (event_attr));
    event_attr.type = PERF_TYPE_HARDWARE;
    event_attr.size = sizeof (event_attr);
    event_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
    event_attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID | PERF_FORMAT_GROUP;
    event_attr.disabled = 1;
    event_attr.pinned = 1;
    thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS] =
            perf_event_open (&event_attr, 0, cpu, -1, 0);
    mxAssertS (thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS] != -1, "perf_event_open EVENT_HW_INSTRUCTIONS");

    /* Open the other events in the group */
    event_attr.disabled = 0;
    event_attr.pinned = 0;
    event_attr.type = PERF_TYPE_HARDWARE;
    event_attr.config = PERF_COUNT_HW_REF_CPU_CYCLES;
    thread_data->perf_event_fds[EVENT_HW_REF_CPU_CYCLES] =
            perf_event_open (&event_attr, 0, cpu, thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS], 0);
    if (thread_data->perf_event_fds[EVENT_HW_REF_CPU_CYCLES] == -1)
    {
        mexErrMsgIdAndTxt ("time_matrix_multiply:perf_event_open",
                "EVENT_HW_REF_CPU_CYCLES errno=%d, error=%s", errno, strerror (errno));
    }
    mxAssertS (thread_data->perf_event_fds[EVENT_HW_REF_CPU_CYCLES] != -1, "perf_event_open EVENT_HW_REF_CPU_CYCLES");

    event_attr.type = PERF_TYPE_HW_CACHE;
    event_attr.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    thread_data->perf_event_fds[EVENT_L1D_READ_ACCESS] =
            perf_event_open (&event_attr, 0, cpu, thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS], 0);
    mxAssertS (thread_data->perf_event_fds[EVENT_L1D_READ_ACCESS] != -1, "perf_event_open EVENT_L1D_READ_ACCESS");

    event_attr.type = PERF_TYPE_HW_CACHE;
    event_attr.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    thread_data->perf_event_fds[EVENT_L1D_READ_MISS] =
            perf_event_open (&event_attr, 0, cpu, thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS], 0);
    mxAssertS (thread_data->perf_event_fds[EVENT_L1D_READ_MISS] != -1, "perf_event_open EVENT_L1D_READ_MISS");

    event_attr.type = PERF_TYPE_HW_CACHE;
    event_attr.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    thread_data->perf_event_fds[EVENT_L1D_WRITE_ACCESS] =
            perf_event_open (&event_attr, 0, cpu, thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS], 0);
    mxAssertS (thread_data->perf_event_fds[EVENT_L1D_WRITE_ACCESS] != -1, "perf_event_open EVENT_L1D_WRITE_ACCESS");

    event_attr.type = PERF_TYPE_HW_CACHE;
    event_attr.config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    thread_data->perf_event_fds[EVENT_L1D_WRITE_MISS] =
            perf_event_open (&event_attr, 0, cpu, thread_data->perf_event_fds[EVENT_HW_INSTRUCTIONS], 0);
    mxAssertS (thread_data->perf_event_fds[EVENT_L1D_WRITE_MISS] != -1, "perf_event_open EVENT_L1D_WRITE_MISS");
}    

/* Close the perf event file descriptors */
static void close_perf_events (timed_thread_data *const thread_data)
{
    int rc;
    hw_perf_events event_index;

    for (event_index = 0; event_index < NUM_HW_PERF_EVENTS; event_index++)
    {
        rc = close (thread_data->perf_event_fds[event_index]);
        mxAssertS (rc == 0, "close perf_event_fds");
    }
}

/* Reset and enable all perf events, by operating on the group leader */
static void reset_and_enable_perf_events (timed_thread_data *const thread_data)
{
    int rc;
    
    rc = ioctl (thread_data->perf_event_fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    mxAssertS (rc == 0, "PERF_EVENT_IOC_RESET");
    rc = ioctl (thread_data->perf_event_fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    mxAssertS (rc == 0, "PERF_EVENT_IOC_ENABLE");
}

/* Disable and then read the values of all perf events. */
static void disable_and_read_perf_events (timed_thread_data *const thread_data)
{
    int rc;
    hw_events_group_format perf_group;
    ssize_t bytes_read;
    hw_perf_events event_index;
    hw_perf_events id_index;
    SAL_ui64 event_id;
    bool id_found;

    rc = ioctl (thread_data->perf_event_fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    mxAssertS (rc == 0, "PERF_EVENT_IOC_DISABLE");
    
    bytes_read = read (thread_data->perf_event_fds[0], &perf_group, sizeof (perf_group));
    mxAssertS (bytes_read == sizeof (perf_group), "read (hw_events_group_format)");
    mxAssertS (perf_group.num_events == NUM_HW_PERF_EVENTS, "perf_group.num_events");

    /* Extract the event values from the group based upon event IDs, rather than
     * assuming the perf_group.events[] array is in the same order as the 
     * hw_perf_events enumeration. */
    for (event_index = 0; event_index < NUM_HW_PERF_EVENTS; event_index++)
    {
        rc = ioctl (thread_data->perf_event_fds[event_index], PERF_EVENT_IOC_ID, &event_id);
        mxAssertS (rc == 0, "PERF_EVENT_IOC_ID");
        
        id_found = false;
        for (id_index = 0; !id_found && (id_index < NUM_HW_PERF_EVENTS); id_index++)
        {
            if (perf_group.events[id_index].id == event_id)
            {
                thread_data->perf_event_values[event_index] = perf_group.events[id_index].value;
                id_found = true;
            }
        }
        mxAssertS (id_found, "perf event_id not found");
    }
}

static void *timing_thread (void *arg)
{
    timed_thread_data *const thread_data = (timed_thread_data *) arg;
    SAL_i32 warmup_iter;
    SAL_i32 timed_iter;
    struct timespec start_time, stop_time;
    SAL_i32 blocking_thread_index;
    int rc;

    open_perf_events (thread_data);
        
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
    
    rc = getrusage (RUSAGE_SELF, &thread_data->start_self_usage);
    mxAssertS (rc == 0, "getrusage (RUSAGE_SELF)");
    rc = getrusage (RUSAGE_THREAD, &thread_data->start_thread_usage);
    mxAssertS (rc == 0, "getrusage (RUSAGE_THREAD)");
    reset_and_enable_perf_events (thread_data);
    
    for (timed_iter = 0; timed_iter < thread_data->num_timed_iterations; timed_iter++)
    {
        thread_data->times[timed_iter].start_outer_rdtsc = read_rdtsc();
        clock_gettime (CLOCK_MONOTONIC_RAW, &thread_data->times[timed_iter].start_time);
        thread_data->times[timed_iter].start_inner_rdtsc = read_rdtsc();
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        thread_data->times[timed_iter].stop_inner_rdtsc = read_rdtsc();
        clock_gettime (CLOCK_MONOTONIC_RAW, &thread_data->times[timed_iter].stop_time);
        thread_data->times[timed_iter].stop_outer_rdtsc = read_rdtsc();
    }
    
    disable_and_read_perf_events (thread_data);
    rc = getrusage (RUSAGE_SELF, &thread_data->stop_self_usage);
    mxAssertS (rc == 0, "getrusage (RUSAGE_SELF)");
    rc = getrusage (RUSAGE_THREAD, &thread_data->stop_thread_usage);
    mxAssertS (rc == 0, "getrusage (RUSAGE_THREAD)");
    
    /* Tell the blocking threads to exit */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        thread_data->blocking_sems[blocking_thread_index].exit_spin_loop = true;
    }

    close_perf_events (thread_data);
    
    return NULL;
}

static mxArray *create_long_scalar (const long value)
{
    mxArray *const array = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL);
    long *const array_data = mxGetData (array);
    
    *array_data = value;
    return array;
}

static mxArray *create_ulong_scalar (const SAL_ui64 value)
{
    mxArray *const array = mxCreateNumericMatrix (1, 1, mxUINT64_CLASS, mxREAL);
    SAL_ui64 *const array_data = mxGetData (array);
    
    *array_data = value;
    return array;
}

static mxArray *create_rusage_struct (const struct rusage *const usage)
{
    const char *field_names[] =
    {
        "ru_utime_tv_sec", "ru_utime_tv_usec", /* user time used */
        "ru_stime_tv_sec", "ru_stime_tv_usec", /* system time used */
        "ru_maxrss",                           /* maximum resident set size */
        "ru_minflt",                           /* page reclaims */
        "ru_majflt",                           /* page faults */
        "ru_nvcsw",                            /* voluntary context switches */
        "ru_nivcsw"                            /* involuntary context switches */
    };
    mxArray *const rusage_struct = mxCreateStructMatrix (1, 1, 9, field_names);
    
    mxSetField (rusage_struct, 0, "ru_utime_tv_sec", create_long_scalar (usage->ru_utime.tv_sec));
    mxSetField (rusage_struct, 0, "ru_utime_tv_usec", create_long_scalar (usage->ru_utime.tv_usec));
    mxSetField (rusage_struct, 0, "ru_stime_tv_sec", create_long_scalar (usage->ru_stime.tv_sec));
    mxSetField (rusage_struct, 0, "ru_stime_tv_usec", create_long_scalar (usage->ru_stime.tv_usec));
    mxSetField (rusage_struct, 0, "ru_maxrss", create_long_scalar (usage->ru_maxrss));
    mxSetField (rusage_struct, 0, "ru_minflt", create_long_scalar (usage->ru_minflt));
    mxSetField (rusage_struct, 0, "ru_majflt", create_long_scalar (usage->ru_majflt));
    mxSetField (rusage_struct, 0, "ru_nvcsw", create_long_scalar (usage->ru_nvcsw));
    mxSetField (rusage_struct, 0, "ru_nivcsw", create_long_scalar (usage->ru_nivcsw));
    
    return rusage_struct;
}

static mxArray *create_perf_events_struct (const timed_thread_data *const thread_data)
{
    const char *field_names[] =
    {
        "hw_instructions",
        "hw_ref_cpu_cycles",
        "l1d_read_access",
        "l1d_read_miss",
        "l1d_write_access",
        "l1d_write_miss"
    };
    mxArray *const perf_events_struct = mxCreateStructMatrix (1, 1, 6, field_names);

    mxSetField (perf_events_struct, 0, "hw_instructions",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_HW_INSTRUCTIONS]));
    mxSetField (perf_events_struct, 0, "hw_ref_cpu_cycles",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_HW_REF_CPU_CYCLES]));
    mxSetField (perf_events_struct, 0, "l1d_read_access",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_L1D_READ_ACCESS]));
    mxSetField (perf_events_struct, 0, "l1d_read_miss",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_L1D_READ_MISS]));
    mxSetField (perf_events_struct, 0, "l1d_write_access",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_L1D_WRITE_ACCESS]));
    mxSetField (perf_events_struct, 0, "l1d_write_miss",
            create_ulong_scalar (thread_data->perf_event_values[EVENT_L1D_WRITE_MISS]));
            
    return perf_events_struct;
}

mxArray *time_matrix_multiply (void (*matrix_func) (void *),
                               void *test_specific_context,
                               const SAL_i32 num_timed_iterations,
                               const bool block_other_cpus)
{
    timed_thread_data thread_data __attribute__ ((aligned (CACHE_LINE_SIZE)));
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
    mxArray *start_times_outer_rdtsc;
    SAL_ui64 *start_times_outer_rdtsc_data;
    mxArray *stop_times_outer_rdtsc;
    SAL_ui64 *stop_times_outer_rdtsc_data;
    mxArray *start_times_inner_rdtsc;
    SAL_ui64 *start_times_inner_rdtsc_data;
    mxArray *stop_times_inner_rdtsc;
    SAL_ui64 *stop_times_inner_rdtsc_data;
    const char *field_names[] =
    {
        "start_times_tv_sec", "start_times_tv_nsec", "stop_times_tv_sec", "stop_times_tv_nsec",
        "start_times_outer_rdtsc", "stop_times_outer_rdtsc",
        "start_times_inner_rdtsc", "stop_times_inner_rdtsc",
        "start_self_usage", "start_thread_usage", "stop_self_usage", "stop_thread_usage",
        "perf_events"
    };
    
    rc = mlockall (MCL_CURRENT | MCL_FUTURE);
    if (rc != 0)
    {
        mexErrMsgIdAndTxt ("time_matrix_multiply:mlockall",
                "This can fail due to insufficient limits of the amount of locked memory : errno=%d, error=%s", errno, strerror (errno));
    }
    
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
        blocking_sems->exit_spin_loop = false;
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
                "This can fail due to insufficient permission to set rtprio : rc=%d, error=%s", rc, strerror (rc));
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
    start_times_inner_rdtsc = mxCreateNumericMatrix (1, num_timed_iterations, mxUINT64_CLASS, mxREAL);
    stop_times_inner_rdtsc = mxCreateNumericMatrix (1, num_timed_iterations, mxUINT64_CLASS, mxREAL);
    start_times_outer_rdtsc = mxCreateNumericMatrix (1, num_timed_iterations, mxUINT64_CLASS, mxREAL);
    stop_times_outer_rdtsc = mxCreateNumericMatrix (1, num_timed_iterations, mxUINT64_CLASS, mxREAL);
    start_times_tv_sec_data = mxGetData (start_times_tv_sec);
    start_times_tv_nsec_data = mxGetData (start_times_tv_nsec);
    stop_times_tv_sec_data = mxGetData (stop_times_tv_sec);
    stop_times_tv_nsec_data = mxGetData (stop_times_tv_nsec);
    start_times_inner_rdtsc_data = mxGetData (start_times_inner_rdtsc);
    stop_times_inner_rdtsc_data = mxGetData (stop_times_inner_rdtsc);
    start_times_outer_rdtsc_data = mxGetData (start_times_outer_rdtsc);
    stop_times_outer_rdtsc_data = mxGetData (stop_times_outer_rdtsc);
    for (timed_iter = 0; timed_iter < num_timed_iterations; timed_iter++)
    {
        start_times_tv_sec_data[timed_iter] = thread_data.times[timed_iter].start_time.tv_sec;
        start_times_tv_nsec_data[timed_iter] = thread_data.times[timed_iter].start_time.tv_nsec;
        stop_times_tv_sec_data[timed_iter] = thread_data.times[timed_iter].stop_time.tv_sec;
        stop_times_tv_nsec_data[timed_iter] = thread_data.times[timed_iter].stop_time.tv_nsec;
        start_times_inner_rdtsc_data[timed_iter] = thread_data.times[timed_iter].start_inner_rdtsc;
        stop_times_inner_rdtsc_data[timed_iter] = thread_data.times[timed_iter].stop_inner_rdtsc;
        start_times_outer_rdtsc_data[timed_iter] = thread_data.times[timed_iter].start_outer_rdtsc;
        stop_times_outer_rdtsc_data[timed_iter] = thread_data.times[timed_iter].stop_outer_rdtsc;
    }
    mxFree (thread_data.times);
    
    timing_results = mxCreateStructMatrix (1, 1, 13, field_names);
    mxSetField (timing_results, 0, "start_times_tv_sec", start_times_tv_sec);
    mxSetField (timing_results, 0, "start_times_tv_nsec", start_times_tv_nsec);
    mxSetField (timing_results, 0, "stop_times_tv_sec", stop_times_tv_sec);
    mxSetField (timing_results, 0, "stop_times_tv_nsec", stop_times_tv_nsec);
    mxSetField (timing_results, 0, "start_times_inner_rdtsc", start_times_inner_rdtsc);
    mxSetField (timing_results, 0, "stop_times_inner_rdtsc", stop_times_inner_rdtsc);
    mxSetField (timing_results, 0, "start_times_outer_rdtsc", start_times_outer_rdtsc);
    mxSetField (timing_results, 0, "stop_times_outer_rdtsc", stop_times_outer_rdtsc);
    mxSetField (timing_results, 0, "start_self_usage", create_rusage_struct (&thread_data.start_self_usage));
    mxSetField (timing_results, 0, "start_thread_usage", create_rusage_struct (&thread_data.start_thread_usage));
    mxSetField (timing_results, 0, "stop_self_usage", create_rusage_struct (&thread_data.stop_self_usage));
    mxSetField (timing_results, 0, "stop_thread_usage", create_rusage_struct (&thread_data.stop_thread_usage));
    mxSetField (timing_results, 0, "perf_events", create_perf_events_struct (&thread_data));

    return timing_results;
}
