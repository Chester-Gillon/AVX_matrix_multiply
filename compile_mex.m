% Compile all mex files with options set to:
% - Optimise
% - Allow AVX intrinsics to be used
% - Save compiler tempory files to allow the generated assembler to be
%   viewed
function compile_mex
    % Ubuntu 16.04 LTS has gcc 5.4.0 installed by default.
    % As of Matlab 2017a the version currently supported by MEX is '4.9.x'.
    % Therefore, if gcc-4.9 is installed use that specific version.
    %
    % The following was done to install gcc-4.9:
    %  sudo apt-get install gcc-4.9
    %  sudo apt-get install g++-4.9
    %
    % g++-4.9 is installed since mex adds stdc++ to the list of libraries.
    mex_preferred_gcc_version='/usr/bin/gcc-4.9';
    if exist (mex_preferred_gcc_version,'file') == 2
        gcc_ver = ['GCC=' mex_preferred_gcc_version];
    else
        gcc_ver = '';
    end

    opensal_root = fullfile(getenv('HOME'),'opensal-1.0.0');
    opensal_include = fullfile(opensal_root,'include');
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS=$CFLAGS -Wall', 'fp_classify_matrix.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_matrix_multiply.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_split_matrix_multiply.c', 'avx_matrix_multiply_library.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_interleave_matrix_multiply.c', 'avx_matrix_multiply_library.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], ['-L' opensal_root], '-lrt', '-lsal64', 'CFLAGS="$CFLAGS -save-temps -Wall"', 'c_opensal_matrix_multiply.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_matrix_multiply.c', 'cmat_mulx_fixed_dimension_matrix_multiplies.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_accumulate_matrix_multiply.c', 'cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_looped_accumulate_matrix_multiply.c', 'cmat_mulx_fixed_dimension_looped_accumulate_matrix_multiplies.c', 'matrix_utils.c')
    mex (gcc_ver, 'COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', '-lrt', 'c_avx_fixed_dimension_stripmine_matrix_multiply.c', 'cmat_mulx_fixed_dimension_stripmine_matrix_multiplies.c', 'matrix_utils.c')
end
