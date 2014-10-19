% Compile all mex files with options set to:
% - Optimise
% - Allow AVX intrinsics to be used
% - Save compiler tempory files to allow the generated assembler to be
%   viewed
function compile_mex
    opensal_root = fullfile(getenv('HOME'),'opensal-1.0.0');
    opensal_include = fullfile(opensal_root,'include');
    mex COPTIMFLAGS=-O3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_matrix_multiply.c
    mex COPTIMFLAGS=-O3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_avx_split_matrix_multiply.c avx_matrix_multiply_library.c
    mex COPTIMFLAGS=-O3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_avx_interleave_matrix_multiply.c avx_matrix_multiply_library.c
    mex ('COPTIMFLAGS=-O3', ['-I' opensal_include], ['-L' opensal_root], '-lsal64', 'CFLAGS="$CFLAGS -save-temps -Wall"', 'c_opensal_matrix_multiply.c')
    mex ('COPTIMFLAGS=-O3', ['-I' opensal_include], 'CFLAGS="$CFLAGS -mavx -save-temps -Wall"', 'c_avx_fixed_dimension_matrix_multiply.c', 'avx_fixed_dimension_matrix_multiply.c')
end

