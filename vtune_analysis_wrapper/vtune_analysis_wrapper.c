/*
 * @file vtune_analysis_wrapper.c
 * @date 6 Dec 2014
 * @author Chester Gillon
 * @brief Executable for calling AVX Matrix Multiply functions so they may be analysed by Intel vtune
 * @details
 *  This program was created since vtune locked-up when attempted to analyse the functions when called from MATLAB
 */

#include <stdbool.h>
#include <semaphore.h>
#include <sys/resource.h>

#include <sal.h>

#include "cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.h"
#include "standalone_matrix_utils.h"

typedef struct
{
    SAL_i32 nr_c, nc_c, dot_product_length;
    SAL_cf32 *left_matrix_rows[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C];
    SAL_cf32 *right_matrix_rows[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH];
    SAL_cf32 *output_matrix_rows[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C];
    cmat_mulx_fixed_dimension_accumulate_func cmat_mulx_func;
} matrix_context;

static void timed_c_matrix_multiply (void *arg)
{
    matrix_context *const context = (matrix_context *) arg;

    (*context->cmat_mulx_func) (context->left_matrix_rows,
                                context->right_matrix_rows,
                                context->output_matrix_rows,
                                context->nc_c);
}

int main (int argc, char *argv[])
{
    timed_thread_data thread_data __attribute__ ((aligned (CACHE_LINE_SIZE)));
	matrix_context context;
	SAL_i32 num_timed_iterations = 10000;
	bool block_other_cpus = true;

	context.nr_c = 8;
	context.dot_product_length = 8;
	context.nc_c = 57344;

	calloc_row_matrix (context.nr_c, context.dot_product_length, context.left_matrix_rows);
	calloc_row_matrix (context.dot_product_length, context.nc_c, context.right_matrix_rows);
	calloc_row_matrix (context.nr_c, context.nc_c, context.output_matrix_rows);
	context.cmat_mulx_func = cmat_mulx_avx_accumulate_nr_c_8_dot_product_length_8;

	time_matrix_multiply (&thread_data, timed_c_matrix_multiply, &context, num_timed_iterations, block_other_cpus);

	return 0;
}
