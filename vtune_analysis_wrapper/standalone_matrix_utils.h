/*
 * @file standalone_matrix_utils.h
 * @date 7 Dec 2014
 * @author Chester Gillon
 * @brief Utilities for timing matrix functions outside of the Matlab MEX environment
 */

#ifndef STANDALONE_MATRIX_UTILS_H_
#define STANDALONE_MATRIX_UTILS_H_

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
} timed_thread_data;

void assert_str (bool assertion, const char *error_message);
void *calloc_and_touch (size_t nmemb, size_t size);
void calloc_row_matrix (size_t num_rows, size_t num_columns, SAL_cf32 *rows[]);
void time_matrix_multiply (timed_thread_data *const thread_data,
	                       void (*matrix_func) (void *),
                           void *test_specific_context,
                           const SAL_i32 num_timed_iterations,
                           const bool block_other_cpus);

#endif /* STANDALONE_MATRIX_UTILS_H_ */
