% Compile all mex files with options set to:
% - Optimise
% - Allow AVX intrinsics to be used
% - Save compiler tempory files to allow the generated assembler to be
%   viewed
function compile_mex
    % Ubuntu 16.04 LTS has gcc 5.4.0 installed by default.
    % As of Matlab 2018a the version currently supported by MEX is '6.3.x'.
    % Therefore, if gcc-6.3 is installed use that specific version.
    %
    % The following was done to install gcc-6.3:
    %  sudo add-apt-repository ppa:jonathonf/gcc-6.3
    %  sudo apt-get update
    %  sudo apt-get install gcc-6 g++-6
    %
    % Notes:
    % a) g++-6 is installed since mex adds stdc++ to the list of libraries.
    % b) Had to install 6.3 from the Personal Package Archive at
    %    https://launchpad.net/~jonathonf/+archive/ubuntu/gcc-6.3 since
    %    there is no offical Unbuntu package for gcc 6.3
    % c) http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu contains a
    %    conflicting package also named gcc-6 which is for gcc 6.4.
    mex_preferred_gcc_version='/usr/bin/gcc-6';
    if exist (mex_preferred_gcc_version,'file') == 2
        gcc_ver = ['GCC=' mex_preferred_gcc_version];
    else
        gcc_ver = '';
    end

    opensal_root = fullfile(getenv('HOME'),'opensal-1.0.0');
    opensal_include = fullfile(opensal_root,'include');
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS=$CFLAGS -Wall', 'cpu_supports.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS=$CFLAGS -Wall', 'fp_classify_matrix.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_matrix_multiply.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -mfma -save-temps -Wall"', '-lrt', 'c_avx_split_matrix_multiply.c', 'avx_matrix_multiply_library.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -mfma -save-temps -Wall"', '-lrt', 'c_avx_interleave_matrix_multiply.c', 'avx_matrix_multiply_library.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -mfma -save-temps -Wall"', '-lrt', 'c_avx_fma_interleave_matrix_multiply.c', 'avx_matrix_multiply_library.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], ['-L' opensal_root], '-lrt', '-lsal64', 'CFLAGS="$CFLAGS -save-temps -Wall"', 'c_opensal_matrix_multiply.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_matrix_multiply.c', 'cmat_mulx_fixed_dimension_matrix_multiplies.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_accumulate_matrix_multiply.c', 'cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -mfma -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_fma_accumulate_matrix_multiply.c', 'cmat_mulx_fixed_dimension_fma_accumulate_matrix_multiplies.c', 'matrix_utils.c')
end
