% Compile all mex files with options set to:
% - Optimise
% - Allow AVX intrinsics to be used
% - Save compiler tempory files to allow the generated assembler to be
%   viewed
function compile_mex
    mex COPTIMFLAGS=-o3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_matrix_multiply.c
    mex COPTIMFLAGS=-o3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_avx_split_matrix_multiply.c avx_matrix_multiply_library.c
    mex COPTIMFLAGS=-o3 CFLAGS="$CFLAGS -mavx -save-temps -Wall" c_avx_interleave_matrix_multiply.c avx_matrix_multiply_library.c
end

