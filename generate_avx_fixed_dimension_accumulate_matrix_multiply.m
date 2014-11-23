function generate_avx_fixed_dimension_accumulate_matrix_multiply
    c_fid = fopen ('cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.c','wt');
    h_fid = fopen ('cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.h','wt');

    lines = { ...
'#include <immintrin.h>' ...
'#include <sal.h>' ...
'' ...
'#include "cmat_mulx_fixed_dimension_accumulate_matrix_multiplies.h"' ...
'' ...
'/* For a complex interleave AVX vector swap the real and imaginary parts */' ...
'#define SWAP_REAL_IMAG_PERMUTE 0xB1'};
    write_lines_to_file (c_fid, lines);
    
    for nr_c = 2:20
        for dot_product_length = 2:20
            generate_function (c_fid, h_fid, nr_c, dot_product_length);
            generated_funcs(nr_c, dot_product_length) = true;
        end
    end
    
    generated_size = size(generated_funcs);
    max_nr_c = generated_size(1);
    max_dot_product_length = generated_size(2);
    lines = { ...
'' ...
'typedef void (*cmat_mulx_fixed_dimension_accumulate_func) (SAL_cf32 *A[],   /* left input matrix array [nr_c] of pointers to each row  */' ...
'                                                           SAL_cf32 *B[],   /* right input matrix array [dot_product_length] of pointers to each row */' ...
'                                                           SAL_cf32 *C[],   /* output matrix array [nr_c] of pointers to each row */' ...
'                                                           SAL_i32 nc_c);   /* column count in C */' ...
'' ...
sprintf('#define CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C %u', max_nr_c) ...
sprintf('#define CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH %u', max_dot_product_length) ...
'extern const cmat_mulx_fixed_dimension_accumulate_func cmat_mulx_fixed_dimension_accumulate_functions[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH+1];'};
    write_lines_to_file (h_fid, lines);

    lines = { ...
'' ...
'const cmat_mulx_fixed_dimension_accumulate_func cmat_mulx_fixed_dimension_accumulate_functions[CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_ACCUMULATE_MAX_DOT_PRODUCT_LENGTH+1] =' ...
'{'};
    write_lines_to_file (c_fid, lines);
    for nr_c = 0:max_nr_c
        fprintf (c_fid, '    {');
        for dot_product_length = 0:max_dot_product_length
            func_exists = false;
            if (nr_c > 0) && (dot_product_length > 0)
                func_exists = generated_funcs (nr_c, dot_product_length);
            end
            if dot_product_length > 0
                fprintf (c_fid, ',');
            end
            if func_exists
                fprintf (c_fid, 'cmat_mulx_avx_accumulate_nr_c_%u_dot_product_length_%u', nr_c, dot_product_length);
            else
                fprintf (c_fid, 'NULL');
            end
        end
        if nr_c < max_nr_c
            fprintf (c_fid, '},\n');
        else
            fprintf (c_fid, '}\n');
        end
    end
    
    fprintf (c_fid, '};\n');
    
    fclose (c_fid);
    fclose (h_fid);
end

function generate_function (c_fid, h_fid, nr_c, dot_product_length)
    func_name = sprintf ('cmat_mulx_avx_accumulate_nr_c_%u_dot_product_length_%u', ...
        nr_c, dot_product_length);
    % Write function definition / prototype
    func_type_and_name = ['void ' func_name ' ('];
    nr_c_array = sprintf('%-6s', sprintf('[%u],', nr_c));
    dot_product_array = sprintf('%-6s', sprintf('[%u],', dot_product_length));
    align_spaces = repmat (' ', [1 length(func_type_and_name)]);
    lines = { ...
[func_type_and_name 'SAL_cf32 *A' nr_c_array '/* left input matrix array [nr_c] of pointers to each row */'] ...
[align_spaces       'SAL_cf32 *B' dot_product_array '/* right input matrix array [dot_product_length] of pointers to each row */'] ...
[align_spaces       'SAL_cf32 *C' nr_c_array '/* output matrix array [nr_c] of pointers to each row */'] ...
[align_spaces       'SAL_i32 nc_c)    /* column count in C */']};

    write_lines_to_file (c_fid, lines);
    lines = strrep (lines,') ',');');
    write_lines_to_file (h_fid, lines);
    fprintf (h_fid, '\n');
    
    fprintf (c_fid, '{\n');
    
    % Write variables to hold the left matrix cached in AVX vectors
    for row = 0:nr_c-1
        for r_i=['r' 'i']
            line = '    __m256 ';
            for col = 0:dot_product_length-1
                if col > 0
                    line = [line ', '];
                end
                line = [line sprintf('left_r%u_c%u_%s', row, col, r_i)];
            end
            line = [line ';'];
            fprintf (c_fid, '%s\n', line);
        end
    end
    
    % Write variables to hold one AVX vector from the right matrix
    fprintf (c_fid, '    __m256 right_r_i, right_i_r;\n');

    % Write variables to accumulate one column of the output matrix in AVX
    % vectors
    for row = 0:nr_c-1
        fprintf (c_fid, '    __m256 output_r%u;\n', row);
    end
    
    % Column loop variable
    fprintf (c_fid, '    SAL_i32 c_c;\n');
    
    % Write code to load the left matrix into AVX vectors
    for row = 0:nr_c-1
        fprintf (c_fid, '\n');
        for col = 0:dot_product_length-1
            fprintf (c_fid, '    left_r%u_c%u_r = _mm256_broadcast_ss (&(A[%u] + %u)->real);\n', ...
                row, col, row, col);
            fprintf (c_fid, '    left_r%u_c%u_i = _mm256_broadcast_ss (&(A[%u] + %u)->imag);\n', ...
                row, col, row, col);
        end
    end
    
    % Write start of loop to process one right and output column at a time
    lines = { ...
'' ...    
'    for (c_c = 0; c_c < nc_c; c_c += 4)' ...
'    {'};
    write_lines_to_file (c_fid, lines);

    % Write code to compute one output column.
    % The right matrix is read one AVX vector at a time, and used to
    % accumulate the output column.
    for right_row = 0:dot_product_length-1
        fprintf(c_fid, '        right_r_i = _mm256_load_ps (&(B[%u] + c_c)->real);\n',right_row);
        fprintf(c_fid, '        right_i_r = _mm256_permute_ps (right_r_i, SWAP_REAL_IMAG_PERMUTE);\n');
        for output_row = 0:nr_c-1
            output_row_padding = repmat(' ',[1 length(num2str(output_row))]);
            if right_row == 0
                fprintf(c_fid, '        output_r%u = _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r%u_c%u_r),\n',output_row,output_row,right_row);
                fprintf(c_fid, '                %s                     _mm256_mul_ps (right_i_r, left_r%u_c%u_i));\n',output_row_padding,output_row,right_row);
            elseif right_row < dot_product_length-1
                fprintf(c_fid, '        output_r%u = _mm256_add_ps (output_r%u, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r%u_c%u_r),\n',output_row,output_row,output_row,right_row);
                fprintf(c_fid, '                %s                          %s                    _mm256_mul_ps (right_i_r, left_r%u_c%u_i)));\n',output_row_padding,output_row_padding,output_row,right_row);
            else
                fprintf(c_fid, '        _mm256_store_ps (&(C[%u] + c_c)->real, _mm256_add_ps (output_r%u, _mm256_addsub_ps (_mm256_mul_ps (right_r_i, left_r%u_c%u_r),\n',output_row,output_row,output_row,right_row);
                fprintf(c_fid, '                             %s                                       %s                    _mm256_mul_ps (right_i_r, left_r%u_c%u_i))));\n',output_row_padding,output_row_padding,output_row,right_row);
            end
        end
    end

    % Complete C function
    fprintf (c_fid, '    }\n');
    fprintf (c_fid, '}\n');
    fprintf (c_fid, '\n');
end

function write_lines_to_file (fid, lines)
    for line_num = 1:length(lines)
        fprintf (fid, '%s\n', lines{line_num});
    end
end