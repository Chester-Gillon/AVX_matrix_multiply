AVX_matrix_multiply
===================

Experiment for generating optimised AVX matrix multiply functions.
The data types are 32-bit float complex-split and complex-interleave format.

The software has been developed using Matlab R2014A mex under Linux 64-bit on CentOS 6, using gcc 4.4.

The timing results were generated on a Intel(R) Core(TM) i5-2310 CPU which can run at 2.90GHz,
but with the set_linux_cpu_freq.m function used to fix the CPU frequency to 1800MHz.

The CPU in the test machine:
- Has two channels of memory running at 1333MHz
- 6M of L3 cache, so some of the larger matrix dimensions tested exceed the size of the L3 cache

