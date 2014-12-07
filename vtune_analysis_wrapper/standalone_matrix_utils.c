/*
 * @file standalone_matrix_utils.c
 * @date 7 Dec 2014
 * @author Chester Gillon
 * @brief Utilities for timing matrix functions outside of the Matlab MEX environment
 * @todo This is a modified version of the MEX based matrix_utils.c file. Ideally should be using common code.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <xmmintrin.h>

#include <sal.h>

#include "standalone_matrix_utils.h"

void assert_str (bool assertion, const char *error_message)
{
	if (!assertion)
	{
		fprintf (stderr, "Assertion failed: %s\n", error_message);
		exit (EXIT_FAILURE);
	}
}

static __inline__ SAL_ui64 read_rdtsc (void)
{
    unsigned long lo, hi;
    __asm__ __volatile__ ( "rdtsc" : "=a" (lo), "=d" (hi) );
    return( lo | (hi << 32) );
}

void *calloc_and_touch (size_t nmemb, size_t size)
{
    char *data = calloc (nmemb, size);

    if (data != NULL)
    {
        const size_t total_bytes = nmemb * size;
        const size_t page_size = sysconf (_SC_PAGE_SIZE);
        size_t index;

        for (index = 0; index < total_bytes; index += page_size)
        {
            data[index] = 0;
        }
        data[total_bytes-1] = 0;
    }

    return data;
}

void calloc_row_matrix (size_t num_rows, size_t num_columns, SAL_cf32 *rows[])
{
	const size_t total_columns = (num_columns + 3) & ~3;
	const size_t total_bytes = num_rows * total_columns * sizeof(SAL_cf32);
	int rc;
	SAL_cf32 *matrix;
	SAL_i32 row;

	rc = posix_memalign ((void **) &matrix, CACHE_LINE_SIZE, total_bytes);
	assert_str (rc == 0, "posix_memalign for calloc_row_matrix");

	memset (matrix, 0, total_bytes);

	for (row = 0; row < num_rows; row++)
	{
		rows[row] = matrix + (row * total_columns);
	}
}

static void *cpu_blocking_thread (void *arg)
{
    blocking_thread_sems *const blocking_sems = (blocking_thread_sems *) arg;
    int rc;

    rc = sem_post (&blocking_sems->ready_sem);
    assert_str (rc == 0, "sem_post");

    rc = sem_wait (&blocking_sems->go_sem);
    assert_str (rc == 0, "sem_wait");

    do
    {
        _mm_pause ();
    } while (!blocking_sems->exit_spin_loop);

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
        assert_str (rc == 0, "sem_wait");
    }

    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        rc = sem_post (&thread_data->blocking_sems[blocking_thread_index].go_sem);
        assert_str (rc == 0, "sem_post");
    }

    /* Perform a warm-up run to try and get code / data cached for a more
     * representative comparison between different matrix multiply implementations. */
    for (warmup_iter = 0; warmup_iter < 2; warmup_iter++)
    {
        clock_gettime (CLOCK_MONOTONIC_RAW, &start_time);
        (*thread_data->matrix_func) (thread_data->test_specific_context);
        clock_gettime (CLOCK_MONOTONIC_RAW, &stop_time);
    }

    rc = getrusage (RUSAGE_SELF, &thread_data->start_self_usage);
    assert_str (rc == 0, "getrusage (RUSAGE_SELF)");
    rc = getrusage (RUSAGE_THREAD, &thread_data->start_thread_usage);
    assert_str (rc == 0, "getrusage (RUSAGE_THREAD)");

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

    rc = getrusage (RUSAGE_SELF, &thread_data->stop_self_usage);
    assert_str (rc == 0, "getrusage (RUSAGE_SELF)");
    rc = getrusage (RUSAGE_THREAD, &thread_data->stop_thread_usage);
    assert_str (rc == 0, "getrusage (RUSAGE_THREAD)");

    /* Tell the blocking threads to exit */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        thread_data->blocking_sems[blocking_thread_index].exit_spin_loop = true;
    }

    return NULL;
}

void time_matrix_multiply (timed_thread_data *const thread_data,
	                       void (*matrix_func) (void *),
                           void *test_specific_context,
                           const SAL_i32 num_timed_iterations,
                           const bool block_other_cpus)
{
    pthread_t timing_thread_id;
    pthread_t blocking_thread_ids[MAX_CPUS];
    pthread_attr_t attr;
    void *ret_val;
    int rc;
    cpu_set_t cpu_set;
    struct sched_param param;
    const SAL_i32 num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    SAL_i32 blocking_thread_index;

    rc = mlockall (MCL_CURRENT | MCL_FUTURE);
    assert_str (rc == 0, "mlockall");

    thread_data->matrix_func = matrix_func;
    thread_data->test_specific_context = test_specific_context;
    thread_data->num_timed_iterations = num_timed_iterations;
    thread_data->times = calloc_and_touch (num_timed_iterations, sizeof (test_times));
    thread_data->num_blocked_cpus = block_other_cpus ? num_cpus - 1 : 0;

    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data->blocking_sems[blocking_thread_index];
        rc = sem_init (&blocking_sems->ready_sem, 0, 0);
        assert_str (rc == 0, "sem_init");
        rc = sem_init (&blocking_sems->go_sem, 0, 0);
        assert_str (rc == 0, "sem_init");
        blocking_sems->exit_spin_loop = false;
    }

    rc = pthread_attr_init (&attr);
    assert_str (rc == 0, "pthread_attr_init");

    /* Create threads as real-time to minimise interruptions */
    rc = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
    assert_str (rc == 0, "pthread_attr_setinheritsched");

    rc = pthread_attr_setschedpolicy (&attr, SCHED_FIFO);
    assert_str (rc == 0, "pthread_attr_setschedpolicy");

    param.sched_priority = 49;
    rc = pthread_attr_setschedparam (&attr, &param);
    assert_str (rc == 0, "pthread_attr_setschedparam");

    /* Run thread only on highest numbered CPU - which can be isolated from
     * being used for other processes using tuna */
    assert_str (num_cpus <= MAX_CPUS, "sysconf(_SC_NPROCESSORS_ONLN) larger than MAX_CPUS");
    CPU_ZERO (&cpu_set);
    CPU_SET (num_cpus - 1, &cpu_set);
    rc = pthread_attr_setaffinity_np (&attr, sizeof(cpu_set), &cpu_set);
    assert_str (rc == 0, "pthread_attr_setaffinity_np");

    rc = pthread_create (&timing_thread_id, &attr, timing_thread, thread_data);
    if (rc != 0)
    {
    	fprintf (stderr, "pthread_create failed.\n");
    	fprintf (stderr, "This can fail due to insufficient permission to set rtprio : rc=%d, error=%s", rc, strerror (rc));
    }

    /* Start blocking threads on all CPUs not used for the timing test */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data->blocking_sems[blocking_thread_index];

        CPU_ZERO (&cpu_set);
        CPU_SET (blocking_thread_index, &cpu_set);
        rc = pthread_attr_setaffinity_np (&attr, sizeof(cpu_set), &cpu_set);
        assert_str (rc == 0, "pthread_attr_setaffinity_np");

        rc = pthread_create (&blocking_thread_ids[blocking_thread_index], &attr, cpu_blocking_thread, blocking_sems);
        assert_str (rc == 0, "pthread_create");
    }

    rc = pthread_join (timing_thread_id, &ret_val);
    assert_str (rc == 0, "pthread_join");

    /* Terminate blocking threads */
    for (blocking_thread_index = 0; blocking_thread_index < thread_data->num_blocked_cpus; blocking_thread_index++)
    {
        blocking_thread_sems *const blocking_sems = &thread_data->blocking_sems[blocking_thread_index];

        rc = pthread_join (blocking_thread_ids[blocking_thread_index], &ret_val);
        assert_str (rc == 0, "pthread_join");

        rc = sem_destroy (&blocking_sems->ready_sem);
        assert_str (rc == 0, "sem_destroy");
        rc = sem_destroy (&blocking_sems->go_sem);
        assert_str (rc == 0, "sem_destroy");
    }

    rc = pthread_attr_destroy (&attr);
    assert_str (rc == 0, "pthread_attr_destroy");
}
