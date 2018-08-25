AVX_matrix_multiply
===================

Experiment for generating optimised AVX and FMA matrix multiply functions.
The data types are 32-bit float complex-split and complex-interleave format.

The software has been developed using Matlab R2018A mex under Linux 64-bit on Ubuntu 16.04, using gcc 6.3.


Optimisation notes
==================

1) Adding __restrict__ to the cmat_mulx_avx functions didn't change the generated code (as reported by the IACA analysis)


2) The output from compare_compiler_optimisations for different GCC versions:
4.8.5 to 4.9.3 cmat_mulx_avx Min -2.94 Max 6.38 Mean 0.58 Min at nr_c=6,dot_product_length=3 Max at 2,18
4.8.5 to 4.9.3 cmat_mulx_avx_accumulate Min -5.13 Max 1.95 Mean -0.04 Min at nr_c=10,dot_product_length=2 Max at 19,2
4.8.5 to 4.9.3 cmat_mulx_avx_fma_accumulate Min -4.63 Max 13.04 Mean 2.15 Min at nr_c=3,dot_product_length=17 Max at 20,20
4.9.3 to 6.3.1 cmat_mulx_avx Min -16.23 Max 3.42 Mean -4.08 Min at nr_c=3,dot_product_length=20 Max at 10,6
4.9.3 to 6.3.1 cmat_mulx_avx_accumulate Min -28.98 Max 4.10 Mean -18.89 Min at nr_c=10,dot_product_length=20 Max at 17,2
4.9.3 to 6.3.1 cmat_mulx_avx_fma_accumulate Min -11.54 Max 6.94 Mean -2.30 Min at nr_c=20,dot_product_length=20 Max at 9,3
4.8.5 to 6.3.1 cmat_mulx_avx Min -14.58 Max 3.45 Mean -3.56 Min at nr_c=18,dot_product_length=20 Max at 3,5
4.8.5 to 6.3.1 cmat_mulx_avx_accumulate Min -29.15 Max 3.64 Mean -18.94 Min at nr_c=19,dot_product_length=20 Max at 17,2
4.8.5 to 6.3.1 cmat_mulx_avx_fma_accumulate Min -5.72 Max 6.94 Mean -0.36 Min at nr_c=4,dot_product_length=12 Max at 9,3

From the plots looking for trends:
- For cmat_mulx_avx_fma_accumulate from 4.8.5 to 4.9.3 there is a speedup as n_cr increases.
- For cmat_mulx_avx_fma_accumulate from 4.9.3 to 6.3.1 the speedup from 4.8.5 to 4.9.3 has been removed.
- For cmat_mulx_avx_fma_accumulate from 4.8.5 to 6.3.1 there is no overall trend.

- For cmat_mulx_avx_accumulate from 4.8.5 to 6.3.1 there is a slowdown as dot_product_length increases.
- For cmat_mulx_avx_accumulate from 4.8.5 to 4.9.3 there is no overall trend.

- For cmat_mulx_avx from 4.8.5 to 6.3.1 there is a slowdown as dot_product_length increases.
- For cmat_mulx_avx from 4.8.5 to 4.9.3 there is no overall trend.

In summary going from 4.8.5 to 4.9.3 in thoery generates better optimisations,
but going from 4.9.6 to 6.3.1 generates worse optimisations.


3) The output from compare_multiply_types for different GCC versions:
4.8.5 cmat_mulx_avx_accumulate -> cmat_mulx_avx_fma_accumulate Min -25.18 Max 78.66 Mean 42.76 Min at nr_c=2,dot_product_length=20 Max at 14,20
4.9.3 cmat_mulx_avx_accumulate -> cmat_mulx_avx_fma_accumulate Min -25.47 Max 78.66 Mean 45.93 Min at nr_c=2,dot_product_length=20 Max at 14,20
6.3.1 cmat_mulx_avx_accumulate -> cmat_mulx_avx_fma_accumulate Min -1.50 Max 149.36 Mean 78.53 Min at nr_c=2,dot_product_length=20 Max at 14,20
4.8.5 cmat_mulx_avx -> cmat_mulx_avx_accumulate Min -13.25 Max 41.77 Mean 5.15 Min at nr_c=19,dot_product_length=2 Max at 2,20
4.9.3 cmat_mulx_avx -> cmat_mulx_avx_accumulate Min -13.51 Max 34.67 Mean 4.44 Min at nr_c=18,dot_product_length=2 Max at 2,19
6.3.1 cmat_mulx_avx -> cmat_mulx_avx_accumulate Min -24.75 Max 21.21 Mean -11.82 Min at nr_c=20,dot_product_length=13 Max at 2,8
4.8.5 cmat_mulx_avx -> cmat_mulx_avx_fma_accumulate Min 4.41 Max 87.72 Mean 49.05 Min at nr_c=20,dot_product_length=2 Max at 10,20
4.9.3 cmat_mulx_avx -> cmat_mulx_avx_fma_accumulate Min 0.00 Max 85.61 Mean 51.33 Min at nr_c=2,dot_product_length=20 Max at 12,20
6.3.1 cmat_mulx_avx -> cmat_mulx_avx_fma_accumulate Min 4.41 Max 118.15 Mean 54.55 Min at nr_c=20,dot_product_length=2 Max at 14,20

In summary:
- For all compilers cmat_mulx_avx_fma_accumulate is faster than (or the same as) cmat_mulx_avx.
- For all compilers cmat_mulx_avx is faster or slower than cmat_mulx_avx_accumulate for some matrix sizes.
  For 6.3.1, on average cmat_mulx_avx is faster than cmat_mulx_avx_accumulate.
  Whereas for 4.8.5 and 4.9.3, on average cmat_mulx_avx is slower than cmat_mulx_avx_accumulate. 



28-Jan-2018 timing runs
=======================

These timing runs on a Kaby Lake based laptop running Ubuntu 16.04

The following results were from Matlab running in the GUI, dynamic CPU frequency and no specific CPU isolation:
20182801T150324_i5-7200U_3100MHz_matrix_test.csv
20182801T152041_i5-7200U_600-1704MHz_matrix_test.csv
20182801T184209_i5-7200U_3100MHz_matrix_test.csv

The following results were from Matlab running from the command line in run level 1, CPU frequency fixed at 1.9 GHz and no CPU isolation:
20182801T154347_i5-7200U_1900MHz_matrix_test.csv
20182801T160534_i5-7200U_1900MHz_matrix_test.csv
20182801T181432_i5-7200U_1900MHz_matrix_test.csv 

The command for running Matlab from the command line was:
/usr/local/MATLAB/R2017a/bin/matlab -nojvm -nodisplay -nosplash

Looking at which matrix multiply function was fastest across all data sizes when Matlab running in the GUI with frequency scaling:
>> compare_matrix_test_results

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T150324_i5-7200U_3100MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 809
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 406
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 195
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 80
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 49
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 37
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 582
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 296
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 104
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 60
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 21
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 20
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 610
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 352
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 132
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 94
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 65
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 29
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 490
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 244
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 233
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 79
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 40
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 27

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T152041_i5-7200U_600-1704MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 857
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 375
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 188
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 74
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 43
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 39
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 612
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 275
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 89
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 54
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 28
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 25
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 624
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 317
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 147
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 101
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 66
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 27
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 406
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 382
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 190
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 73
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 38
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 24

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T184209_i5-7200U_3100MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 845
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 408
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 172
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 65
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 45
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 41
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 589
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 293
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 100
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 64
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 20
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 17
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 610
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 351
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 129
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 94
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 62
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 36
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 540
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 238
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 197
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 56
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 48
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 34


Looking at which matrix multiply function was fastest across all data sizes when Matlab running in the command line with fixed frequencies:
>> compare_matrix_test_results

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T154347_i5-7200U_1900MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 1094
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 367
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 100
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 7
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 5
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 3
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 688
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 303
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 62
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 22
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 4
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 4
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 678
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 370
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 113
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 66
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 38
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 17
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 558
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 234
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 217
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 48
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 33
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 23

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T160534_i5-7200U_1900MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 1090
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 367
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 102
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 9
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 4
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 4
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 698
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 293
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 58
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 17
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 13
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 4
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 707
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 345
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 101
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 63
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 49
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 17
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 557
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 240
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 208
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 47
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 31
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 30

Processing /home/mr_halfword/AVX_matrix_multiply/20182801T181432_i5-7200U_1900MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 1072
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 392
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 102
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 5
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 3
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 2
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 689
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 297
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 66
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 22
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 7
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 2
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 672
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 374
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 100
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 65
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 56
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 15
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 523
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 286
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 196
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 52
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 33
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 23

The summary is:
a) c_avx_fixed_dimension_fma_accumulate_matrix_multiply, c_avx_fixed_dimension_matrix_multiply or c_avx_fixed_dimension_fma_accumulate_matrix_multiply
   may be the fastest on any given run/combination.

b) c_avx_fixed_dimension_fma_accumulate_matrix_multiply is the fastest function as a percentage over all runs/combinations.


04-Feb-2018 timing runs
=======================

These timing runs on a Kaby Lake based laptop running Ubuntu 16.04, collecting cache PMU events

The following results were from Matlab running from the command line in run level 1, CPU frequency fixed at 1.9 GHz and no CPU isolation:
20180204T203638_i5-7200U_1900MHz_matrix_test.csv

>> compare_matrix_test_results

Processing /home/mr_halfword/AVX_matrix_multiply/20180204T203638_i5-7200U_1900MHz_matrix_test.csv
Data set fits in cache: L1
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 1117
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 351
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 101
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 5
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 2
Data set fits in cache: L2
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 689
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 299
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 63
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 21
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 10
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 1
Data set fits in cache: L3
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 655
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 373
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 115
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 68
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 46
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 25
Data set fits in cache: None
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 536
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 259
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 214
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 55
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 28
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 21


25-08-2018 timing runs
======================

Running on a HP Z640 with dual Xeon E5-2620 v3, with two 8GB ECC RDDR4 1866MHz DIMMs per processor.
Note that the code didn't set an explicit NUMA node for allocations (the code was originally
written for a uniprocessor system).

Test steps:
1. When booting Ubuntu 16.04 LTS with Kernel 4.4.0-133-generic #159-Ubuntu SMP added the following
command line options:
   intel_pstate=disable 1

The first option disable the Intel pstate driver which performs dynamic frequency scaling.

The second option boots into runlevel 1 - command line, single user, no networking

2. As root allocate space for hugepages:
echo 50 > /proc/sys/vm/nr_hugepages

3. As a normal user:
ulimit -l unlimited -r 50
cd AVX_matrix_multiply/
/usr/local/MATLAB/R2018a/bin/matlab -nojvm -nodisplay -nosplash

4. In Matlab:
set_linux_cpu_freq (1800000)
matrix_test

The results are in 20180825T122901_E5-2620_v3_1800MHz_matrix_test.csv


>> compare_matrix_test_results

Processing /home/mr_halfword/AVX_matrix_multiply/20180825T122901_E5-2620_v3_1800MHz_matrix_test.csv
Data set fits in cache: L1  Matrix Allocation Method: avx_align
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 970
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 398
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 70
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 56
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 41
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 41
Data set fits in cache: L2  Matrix Allocation Method: avx_align
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 711
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 276
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 31
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 27
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 21
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 17
Data set fits in cache: L3  Matrix Allocation Method: avx_align
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 1431
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 361
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 238
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 50
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 21
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 17
Data set fits in cache: None  Matrix Allocation Method: avx_align
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 198
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 58
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 11
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 10
Data set fits in cache: L1  Matrix Allocation Method: l1d_stride
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 1056
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 285
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 153
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 39
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 25
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 18
Data set fits in cache: L2  Matrix Allocation Method: l1d_stride
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 852
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 136
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 65
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 15
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 9
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 6
Data set fits in cache: L3  Matrix Allocation Method: l1d_stride
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 1698
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 278
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 114
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 20
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 5
 c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 3
Data set fits in cache: None  Matrix Allocation Method: l1d_stride
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 213
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply : 33
 c_avx_fixed_dimension_fma_accumulate_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_matrix_multiply : 26
 c_avx_fixed_dimension_matrix_multiply > c_avx_fixed_dimension_accumulate_matrix_multiply > c_avx_fixed_dimension_fma_accumulate_matrix_multiply : 5



Notes for old timing results
============================

The timing results were generated on a Intel(R) Core(TM) i5-2310 CPU which can run at 2.90GHz,
but with the set_linux_cpu_freq.m function used to fix the CPU frequency to 1800MHz.

To minimise disruptions from other processes:
- The timing tests were run with CentOS at run level 1, i.e. single user with no network
- MATLAB was started with just a command line and no JVM, i.e.:
    /usr/local/MATLAB/R2014a/bin/matlab -nojvm -nodisplay -nosplash
- The functions which were timed were run on a thread pinned to CPU 3, which had been isolated using tuna.
  i.e. Linux was prevented scheduling other processes on CPU 3.
- The thread timing the function under test was at real time priority 49, with mlockall used

In the timing results with "Block Other CPUs" of one, while CPU 3 was timing the functions under test, the
other CPUs 0 to 2 were kept busy with thread at a real time priority of 49 in a busy loop waiting for the
test to complete. The idea was to prevent CPUs 0 to 2 from running code which could use up L3 / memory
bandwidth while CPU 3 was performing the timing test. For reasons which are not understood, this caused
some timed iterations to take an excessive time, e.g. median duration of 1048.7us but a maximum of 146736.2us.
The time recorded by both clock_gettime (CLOCK_MONOTONIC_RAW) and a raw rdtsc instruction both report an
execssive time, so looks a valid measured time.

The CPU in the test machine:
- Has two channels of memory running at 1333MHz
- 6M of L3 cache, so some of the larger matrix dimensions tested exceed the size of the L3 cache

