AVX_matrix_multiply
===================

Experiment for generating optimised AVX matrix multiply functions.
The data types are 32-bit float complex-split and complex-interleave format.

The software has been developed using Matlab R2014A mex under Linux 64-bit on CentOS 6.6, using gcc 4.4.

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

