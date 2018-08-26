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
    % GCC 6.3.1 was installed from gnat-gpl-2017-x86_64-linux-bin.tar.gz
    % downloaded from https://www.adacore.com/download/more. The install
    % root was set to /opt/GNAT/2017.
    %
    % Note that while https://launchpad.net/~jonathonf/+archive/ubuntu/gcc-6.3
    % has gcc-6.3 packaged for Ubuntu, that package can't be installed on a
    % machine with Beyond Compare 4. due to a conflict on the version of
    % gcc-6-base.
    mex_preferred_gcc_version='/opt/GNAT/2017/bin/gcc';
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
