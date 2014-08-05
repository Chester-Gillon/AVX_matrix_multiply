
/* @todo
 * Initially coded using the Intel vector types and intrinsics as described in:
 *   https://software.intel.com/sites/landingpage/IntrinsicsGuide/
 *
 * When looking at the generated assembler for packed_8_split_matrix_multiply() and 
 * packed_8_interleave_matrix_multiply() noticed that the compiler appeared to be inserting
 * copies into and out of the stack for intermediate results even when compiled with optimisation.
 * i.e. wasn't making best use of registers.
 *
 * The GCC avxintrin.h defines the __m256 type with an attribute of __may_alias__ with the comment:
 *   "The Intel API is flexible enough that we must allow aliasing with other
 *    vector types, and their scalar components."
 *
 * The packed_8_interleave_matrix_multiply_no_alias function is a version which uses the 
 * non-aliased type __v8sf and the GCC intrinsics described in:
 *   https://gcc.gnu.org/onlinedocs/gcc-4.9.1/gcc/X86-Built-in-Functions.html
 *
 * Changing to the non-alised type appears to have reduced the number of uncessary copies
 * to/from the stack, although have yet to time the differences.
 * Still looks to be address calculations inside each loop.
 *
 *
 * http://www.thinkingandcomputing.com/2014/02/28/using-avx-instructions-in-matrix-multiplication/ is
 * an example optimisation which has variables named after the vector registers and re-uses the
 * variables in an attempt to maximise register usage?
 */

#include <immintrin.h>

#include "avx_matrix_multiply_library.h"

#define MAX_SETS 8

void packed_8_split_matrix_multiply (const packed_8_split *const weights,
                                     const split *const samples,
                                     const split *const outputs,
                                     const size_t num_samples,
                                     const size_t num_sets)
{
    __m256 sample_0_real, sample_0_imag;
    __m256 sample_1_real, sample_1_imag;
    __m256 sample_2_real, sample_2_imag;
    __m256 sample_3_real, sample_3_imag;
    __m256 sample_4_real, sample_4_imag;
    __m256 sample_5_real, sample_5_imag;
    __m256 sample_6_real, sample_6_imag;
    __m256 sample_7_real, sample_7_imag;
    __m256 weights_0_real[MAX_SETS], weights_0_imag[MAX_SETS];
    __m256 weights_1_real[MAX_SETS], weights_1_imag[MAX_SETS];
    __m256 weights_2_real[MAX_SETS], weights_2_imag[MAX_SETS];
    __m256 weights_3_real[MAX_SETS], weights_3_imag[MAX_SETS];
    __m256 weights_4_real[MAX_SETS], weights_4_imag[MAX_SETS];
    __m256 weights_5_real[MAX_SETS], weights_5_imag[MAX_SETS];
    __m256 weights_6_real[MAX_SETS], weights_6_imag[MAX_SETS];
    __m256 weights_7_real[MAX_SETS], weights_7_imag[MAX_SETS];
    size_t sample_index;
    size_t set_index;
    
    for (set_index = 0; set_index < num_sets; set_index++)
    {
        weights_0_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[0]);
        weights_0_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[0]);
        weights_1_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[1]);
        weights_1_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[1]);
        weights_2_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[2]);
        weights_2_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[2]);
        weights_3_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[3]);
        weights_3_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[3]);
        weights_4_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[4]);
        weights_4_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[4]);
        weights_5_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[5]);
        weights_5_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[5]);
        weights_6_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[6]);
        weights_6_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[6]);
        weights_7_real[set_index] = _mm256_broadcast_ss (&weights[set_index].real->data[7]);
        weights_7_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].imag->data[7]);
    }
    
    for (sample_index = 0; sample_index < num_samples; sample_index += 8)
    {
        sample_0_real = _mm256_load_ps (&samples[0].real[sample_index]);
        sample_0_imag = _mm256_load_ps (&samples[0].imag[sample_index]);
        sample_1_real = _mm256_load_ps (&samples[1].real[sample_index]);
        sample_1_imag = _mm256_load_ps (&samples[1].imag[sample_index]);
        sample_2_real = _mm256_load_ps (&samples[2].real[sample_index]);
        sample_2_imag = _mm256_load_ps (&samples[2].imag[sample_index]);
        sample_3_real = _mm256_load_ps (&samples[3].real[sample_index]);
        sample_3_imag = _mm256_load_ps (&samples[3].imag[sample_index]);
        sample_4_real = _mm256_load_ps (&samples[4].real[sample_index]);
        sample_4_imag = _mm256_load_ps (&samples[4].imag[sample_index]);
        sample_5_real = _mm256_load_ps (&samples[5].real[sample_index]);
        sample_5_imag = _mm256_load_ps (&samples[5].imag[sample_index]);
        sample_6_real = _mm256_load_ps (&samples[6].real[sample_index]);
        sample_6_imag = _mm256_load_ps (&samples[6].imag[sample_index]);
        sample_7_real = _mm256_load_ps (&samples[7].real[sample_index]);
        sample_7_imag = _mm256_load_ps (&samples[7].imag[sample_index]);
        
        for (set_index = 0; set_index < num_sets; set_index++)
        {
            _mm256_store_ps (&outputs[set_index].real[sample_index],
               _mm256_add_ps (
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (sample_0_real, weights_0_real[set_index]),
                                                  _mm256_mul_ps (sample_0_imag, weights_0_imag[set_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (sample_1_real, weights_1_real[set_index]),
                                                  _mm256_mul_ps (sample_1_imag, weights_1_imag[set_index]))),
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (sample_2_real, weights_2_real[set_index]),
                                                  _mm256_mul_ps (sample_2_imag, weights_2_imag[set_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (sample_3_real, weights_3_real[set_index]),
                                                  _mm256_mul_ps (sample_3_imag, weights_3_imag[set_index])))),
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (sample_4_real, weights_4_real[set_index]),
                                                  _mm256_mul_ps (sample_4_imag, weights_4_imag[set_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (sample_5_real, weights_5_real[set_index]),
                                                  _mm256_mul_ps (sample_5_imag, weights_5_imag[set_index]))),
                    _mm256_add_ps (_mm256_sub_ps (_mm256_mul_ps (sample_6_real, weights_6_real[set_index]),
                                                  _mm256_mul_ps (sample_6_imag, weights_6_imag[set_index])),
                                   _mm256_sub_ps (_mm256_mul_ps (sample_7_real, weights_7_real[set_index]),
                                                  _mm256_mul_ps (sample_7_imag, weights_7_imag[set_index]))))));

            _mm256_store_ps (&outputs[set_index].imag[sample_index],
               _mm256_add_ps (
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (sample_0_imag, weights_0_real[set_index]),
                                                  _mm256_mul_ps (sample_0_real, weights_0_imag[set_index])),
                                   _mm256_add_ps (_mm256_mul_ps (sample_1_imag, weights_1_real[set_index]),
                                                  _mm256_mul_ps (sample_1_real, weights_1_imag[set_index]))),
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (sample_2_imag, weights_2_real[set_index]),
                                                  _mm256_mul_ps (sample_2_real, weights_2_imag[set_index])),
                                   _mm256_add_ps (_mm256_mul_ps (sample_3_imag, weights_3_real[set_index]),
                                                  _mm256_mul_ps (sample_3_real, weights_3_imag[set_index])))),
                 _mm256_add_ps (
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (sample_4_imag, weights_4_real[set_index]),
                                                  _mm256_mul_ps (sample_4_real, weights_4_imag[set_index])),
                                   _mm256_add_ps (_mm256_mul_ps (sample_5_imag, weights_5_real[set_index]),
                                                  _mm256_mul_ps (sample_5_real, weights_5_imag[set_index]))),
                    _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (sample_6_imag, weights_6_real[set_index]),
                                                  _mm256_mul_ps (sample_6_real, weights_6_imag[set_index])),
                                   _mm256_add_ps (_mm256_mul_ps (sample_7_imag, weights_7_real[set_index]),
                                                  _mm256_mul_ps (sample_7_real, weights_7_imag[set_index]))))));
/*                output_real[output_index] +=
                        (samples_real[sample_index] * weights_real[weight_index]) -
                        (samples_imag[sample_index] * weights_imag[weight_index]);
                output_imag[output_index] +=
                        (samples_imag[sample_index] * weights_real[weight_index]) +
                        (samples_real[sample_index] * weights_imag[weight_index]);*/
        }
    }
}

#define SWAP_REAL_IMAG_PERMUTE 0xB1

void packed_8_interleave_matrix_multiply (const packed_8_interleave *const  weights,
                                          const complex *const *const samples,
                                          complex *const *const outputs,
                                          const size_t num_samples,
                                          const size_t num_sets)
{
    __m256 samples_0_real_imag, samples_0_imag_real;
    __m256 samples_1_real_imag, samples_1_imag_real;
    __m256 samples_2_real_imag, samples_2_imag_real;
    __m256 samples_3_real_imag, samples_3_imag_real;
    __m256 samples_4_real_imag, samples_4_imag_real;
    __m256 samples_5_real_imag, samples_5_imag_real;
    __m256 samples_6_real_imag, samples_6_imag_real;
    __m256 samples_7_real_imag, samples_7_imag_real;
    __m256 weights_0_real[MAX_SETS], weights_0_imag[MAX_SETS];
    __m256 weights_1_real[MAX_SETS], weights_1_imag[MAX_SETS];
    __m256 weights_2_real[MAX_SETS], weights_2_imag[MAX_SETS];
    __m256 weights_3_real[MAX_SETS], weights_3_imag[MAX_SETS];
    __m256 weights_4_real[MAX_SETS], weights_4_imag[MAX_SETS];
    __m256 weights_5_real[MAX_SETS], weights_5_imag[MAX_SETS];
    __m256 weights_6_real[MAX_SETS], weights_6_imag[MAX_SETS];
    __m256 weights_7_real[MAX_SETS], weights_7_imag[MAX_SETS];
    size_t sample_index;
    size_t set_index;
    
    for (set_index = 0; set_index < num_sets; set_index++)
    {
        weights_0_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[0].real);
        weights_0_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[0].imag);
        weights_1_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[1].real);
        weights_1_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[1].imag);
        weights_2_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[2].real);
        weights_2_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[2].imag);
        weights_3_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[3].real);
        weights_3_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[3].imag);
        weights_4_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[4].real);
        weights_4_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[4].imag);
        weights_5_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[5].real);
        weights_5_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[5].imag);
        weights_6_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[6].real);
        weights_6_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[6].imag);
        weights_7_real[set_index] = _mm256_broadcast_ss (&weights[set_index].data[7].real);
        weights_7_imag[set_index] = _mm256_broadcast_ss (&weights[set_index].data[7].imag);
    }
    
    for (sample_index = 0; sample_index < num_samples; sample_index += 4)
    {
        samples_0_real_imag = _mm256_load_ps (&samples[0][sample_index].real);
        samples_0_imag_real = _mm256_permute_ps (samples_0_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_1_real_imag = _mm256_load_ps (&samples[1][sample_index].real);
        samples_1_imag_real = _mm256_permute_ps (samples_1_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_2_real_imag = _mm256_load_ps (&samples[2][sample_index].real);
        samples_2_imag_real = _mm256_permute_ps (samples_2_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_3_real_imag = _mm256_load_ps (&samples[3][sample_index].real);
        samples_3_imag_real = _mm256_permute_ps (samples_3_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_4_real_imag = _mm256_load_ps (&samples[4][sample_index].real);
        samples_4_imag_real = _mm256_permute_ps (samples_4_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_5_real_imag = _mm256_load_ps (&samples[5][sample_index].real);
        samples_5_imag_real = _mm256_permute_ps (samples_5_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_6_real_imag = _mm256_load_ps (&samples[6][sample_index].real);
        samples_6_imag_real = _mm256_permute_ps (samples_6_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_7_real_imag = _mm256_load_ps (&samples[7][sample_index].real);
        samples_7_imag_real = _mm256_permute_ps (samples_7_real_imag, SWAP_REAL_IMAG_PERMUTE);
        
        for (set_index = 0; set_index < num_sets; set_index++)
        {
            _mm256_store_ps (&outputs[set_index][sample_index].real,
              _mm256_add_ps(
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (samples_0_real_imag, weights_0_real[set_index]),
                                                      _mm256_mul_ps (samples_0_imag_real, weights_0_imag[set_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (samples_1_real_imag, weights_1_real[set_index]),
                                                      _mm256_mul_ps (samples_1_imag_real, weights_1_imag[set_index]))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (samples_2_real_imag, weights_2_real[set_index]),
                                                      _mm256_mul_ps (samples_2_imag_real, weights_2_imag[set_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (samples_3_real_imag, weights_3_real[set_index]),
                                                      _mm256_mul_ps (samples_3_imag_real, weights_3_imag[set_index])))),
                 _mm256_add_ps (
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (samples_4_real_imag, weights_4_real[set_index]),
                                                      _mm256_mul_ps (samples_4_imag_real, weights_4_imag[set_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (samples_5_real_imag, weights_5_real[set_index]),
                                                      _mm256_mul_ps (samples_5_imag_real, weights_5_imag[set_index]))),
                     _mm256_add_ps (_mm256_addsub_ps (_mm256_mul_ps (samples_6_real_imag, weights_6_real[set_index]),
                                                      _mm256_mul_ps (samples_6_imag_real, weights_6_imag[set_index])),
                                    _mm256_addsub_ps (_mm256_mul_ps (samples_7_real_imag, weights_7_real[set_index]),
                                                      _mm256_mul_ps (samples_7_imag_real, weights_7_imag[set_index]))))));
/*                output_real[output_index] +=
                        (samples_real[sample_index] * weights_real[weight_index]) -
                        (samples_imag[sample_index] * weights_imag[weight_index]);
                output_imag[output_index] +=
                        (samples_imag[sample_index] * weights_real[weight_index]) +
                        (samples_real[sample_index] * weights_imag[weight_index]);*/
        }
    }
}

void packed_8_interleave_matrix_multiply_no_alias (const packed_8_interleave *const  weights,
                                          const complex *const *const samples,
                                          complex *const *const outputs,
                                          const size_t num_samples,
                                          const size_t num_sets)
{
    __v8sf samples_0_real_imag, samples_0_imag_real;
    __v8sf samples_1_real_imag, samples_1_imag_real;
    __v8sf samples_2_real_imag, samples_2_imag_real;
    __v8sf samples_3_real_imag, samples_3_imag_real;
    __v8sf samples_4_real_imag, samples_4_imag_real;
    __v8sf samples_5_real_imag, samples_5_imag_real;
    __v8sf samples_6_real_imag, samples_6_imag_real;
    __v8sf samples_7_real_imag, samples_7_imag_real;
    __v8sf weights_0_real[MAX_SETS], weights_0_imag[MAX_SETS];
    __v8sf weights_1_real[MAX_SETS], weights_1_imag[MAX_SETS];
    __v8sf weights_2_real[MAX_SETS], weights_2_imag[MAX_SETS];
    __v8sf weights_3_real[MAX_SETS], weights_3_imag[MAX_SETS];
    __v8sf weights_4_real[MAX_SETS], weights_4_imag[MAX_SETS];
    __v8sf weights_5_real[MAX_SETS], weights_5_imag[MAX_SETS];
    __v8sf weights_6_real[MAX_SETS], weights_6_imag[MAX_SETS];
    __v8sf weights_7_real[MAX_SETS], weights_7_imag[MAX_SETS];
    size_t sample_index;
    size_t set_index;
    
    for (set_index = 0; set_index < num_sets; set_index++)
    {
        weights_0_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[0].real);
        weights_0_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[0].imag);
        weights_1_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[1].real);
        weights_1_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[1].imag);
        weights_2_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[2].real);
        weights_2_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[2].imag);
        weights_3_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[3].real);
        weights_3_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[3].imag);
        weights_4_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[4].real);
        weights_4_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[4].imag);
        weights_5_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[5].real);
        weights_5_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[5].imag);
        weights_6_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[6].real);
        weights_6_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[6].imag);
        weights_7_real[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[7].real);
        weights_7_imag[set_index] = __builtin_ia32_vbroadcastss256 (&weights[set_index].data[7].imag);
    }
    
    for (sample_index = 0; sample_index < num_samples; sample_index += 4)
    {
        samples_0_real_imag = *(__v8sf *) &samples[0][sample_index].real;
        samples_0_imag_real = __builtin_ia32_vpermilps256 (samples_0_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_1_real_imag = *(__v8sf *) &samples[1][sample_index].real;
        samples_1_imag_real = __builtin_ia32_vpermilps256 (samples_1_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_2_real_imag = *(__v8sf *) &samples[2][sample_index].real;
        samples_2_imag_real = __builtin_ia32_vpermilps256 (samples_2_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_3_real_imag = *(__v8sf *) &samples[3][sample_index].real;
        samples_3_imag_real = __builtin_ia32_vpermilps256 (samples_3_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_4_real_imag = *(__v8sf *) &samples[4][sample_index].real;
        samples_4_imag_real = __builtin_ia32_vpermilps256 (samples_4_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_5_real_imag = *(__v8sf *) &samples[5][sample_index].real;
        samples_5_imag_real = __builtin_ia32_vpermilps256 (samples_5_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_6_real_imag = *(__v8sf *) &samples[6][sample_index].real;
        samples_6_imag_real = __builtin_ia32_vpermilps256 (samples_6_real_imag, SWAP_REAL_IMAG_PERMUTE);
        samples_7_real_imag = *(__v8sf *) &samples[7][sample_index].real;
        samples_7_imag_real = __builtin_ia32_vpermilps256 (samples_7_real_imag, SWAP_REAL_IMAG_PERMUTE);
        
        for (set_index = 0; set_index < num_sets; set_index++)
        {
            _mm256_store_ps (&outputs[set_index][sample_index].real,
              __builtin_ia32_addps256(
                 __builtin_ia32_addps256 (
                     __builtin_ia32_addps256 (__builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_0_real_imag, weights_0_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_0_imag_real, weights_0_imag[set_index])),
                                              __builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_1_real_imag, weights_1_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_1_imag_real, weights_1_imag[set_index]))),
                     __builtin_ia32_addps256 (__builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_2_real_imag, weights_2_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_2_imag_real, weights_2_imag[set_index])),
                                              __builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_3_real_imag, weights_3_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_3_imag_real, weights_3_imag[set_index])))),
                 __builtin_ia32_addps256 (
                     __builtin_ia32_addps256 (__builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_4_real_imag, weights_4_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_4_imag_real, weights_4_imag[set_index])),
                                              __builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_5_real_imag, weights_5_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_5_imag_real, weights_5_imag[set_index]))),
                     __builtin_ia32_addps256 (__builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_6_real_imag, weights_6_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_6_imag_real, weights_6_imag[set_index])),
                                              __builtin_ia32_addsubps256 (__builtin_ia32_mulps256 (samples_7_real_imag, weights_7_real[set_index]),
                                                                          __builtin_ia32_mulps256 (samples_7_imag_real, weights_7_imag[set_index]))))));
/*                output_real[output_index] +=
                        (samples_real[sample_index] * weights_real[weight_index]) -
                        (samples_imag[sample_index] * weights_imag[weight_index]);
                output_imag[output_index] +=
                        (samples_imag[sample_index] * weights_real[weight_index]) +
                        (samples_real[sample_index] * weights_imag[weight_index]);*/
        }
    }
}
