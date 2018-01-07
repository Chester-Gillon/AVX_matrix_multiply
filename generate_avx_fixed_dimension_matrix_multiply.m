function mat_mulx_attribs =generate_avx_fixed_dimension_matrix_multiply (mat_mulx_attribs)
    [~,~] = mkdir ('mat_mul_fragments/cmat_mulx_avx');
    c_fid = fopen ('cmat_mulx_fixed_dimension_matrix_multiplies.c','wt');
    h_fid = fopen ('cmat_mulx_fixed_dimension_matrix_multiplies.h','wt');

    lines = { ...
'#include <immintrin.h>' ...
'#include <sal.h>' ...
'' ...
'#include "cmat_mulx_fixed_dimension_matrix_multiplies.h"' ...
''};
    write_lines_to_file (c_fid, lines);
    
    for nr_c = 2:20
        for dot_product_length = 2:20
            func_name = sprintf ('cmat_mulx_avx_nr_c_%u_dot_product_length_%u', ...
                nr_c, dot_product_length);
            func_filename = sprintf ('mat_mul_fragments/cmat_mulx_avx/%s.c', func_name);
            c_fragment_fid = fopen (func_filename, 'wt');
            num_vfops = generate_function (c_fragment_fid, h_fid, func_name, nr_c, dot_product_length);
            fclose (c_fragment_fid);
            fprintf (c_fid, '#include "%s"\n', func_filename);
            generated_funcs(nr_c, dot_product_length) = true;
            if nargin >= 1
                mat_mulx_attribs{func_name, 'multiply_type'} = {'cmat_mulx_avx'};
                mat_mulx_attribs{func_name, 'nr_c'} = nr_c;
                mat_mulx_attribs{func_name, 'dot_product_length'} = dot_product_length;
                mat_mulx_attribs{func_name, 'num_vfops'} = num_vfops;
            end
        end
    end
    
    generated_size = size(generated_funcs);
    max_nr_c = generated_size(1);
    max_dot_product_length = generated_size(2);
    lines = { ...
'' ...
'typedef void (*cmat_mulx_fixed_dimension_func) (SAL_cf32 *A[],   /* left input matrix array [nr_c] of pointers to each row  */' ...
'                                                SAL_cf32 *B[],   /* right input matrix array [dot_product_length] of pointers to each row */' ...
'                                                SAL_cf32 *C[],   /* output matrix array [nr_c] of pointers to each row */' ...
'                                                SAL_i32 nc_c);   /* column count in C */' ...
'' ...
sprintf('#define CMAT_MULX_FIXED_DIMENSION_MAX_NR_C %u', max_nr_c) ...
sprintf('#define CMAT_MULX_FIXED_DIMENSION_MAX_DOT_PRODUCT_LENGTH %u', max_dot_product_length) ...
'extern const cmat_mulx_fixed_dimension_func cmat_mulx_fixed_dimension_functions[CMAT_MULX_FIXED_DIMENSION_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_MAX_DOT_PRODUCT_LENGTH+1];'};
    write_lines_to_file (h_fid, lines);

    lines = { ...
'' ...
'const cmat_mulx_fixed_dimension_func cmat_mulx_fixed_dimension_functions[CMAT_MULX_FIXED_DIMENSION_MAX_NR_C+1][CMAT_MULX_FIXED_DIMENSION_MAX_DOT_PRODUCT_LENGTH+1] =' ...
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
                fprintf (c_fid, 'cmat_mulx_avx_nr_c_%u_dot_product_length_%u', nr_c, dot_product_length);
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

function num_vfops = generate_function (c_fid, h_fid, func_name, nr_c, dot_product_length)
    lines = {...
'/* For a complex interleave AVX vector swap the real and imaginary parts */' ...
'#ifndef SWAP_REAL_IMAG_PERMUTE' ...
'#define SWAP_REAL_IMAG_PERMUTE 0xB1' ...
'#endif' ...
''};
    write_lines_to_file (c_fid, lines);

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
    
    % Write variables to hold one column of the right matrix in AVX vectors
    for row = 0:dot_product_length-1
        fprintf (c_fid, '    __m256 right_r%u_r_i, right_r%u_i_r;\n', row, row);
    end
    
    % Column loop variable
    fprintf (c_fid, '    SAL_i32 c_c;\n');
    
    % Write code to load the left matrix into AVX vectors
    lines = {...
'#ifdef IACA_LOAD_LEFT' ...
'    IACA_START' ...
'#endif'};
    write_lines_to_file (c_fid, lines);
    for row = 0:nr_c-1
        fprintf (c_fid, '\n');
        for col = 0:dot_product_length-1
            fprintf (c_fid, '    left_r%u_c%u_r = _mm256_broadcast_ss (&(A[%u] + %u)->real);\n', ...
                row, col, row, col);
            fprintf (c_fid, '    left_r%u_c%u_i = _mm256_broadcast_ss (&(A[%u] + %u)->imag);\n', ...
                row, col, row, col);
        end
    end
    lines = {...
'#ifdef IACA_LOAD_LEFT' ...
'    IACA_END' ...
'#endif'};
    write_lines_to_file (c_fid, lines);
    
    % Write start of loop to process one right and output column at a time
    num_vfops = 0;
    lines = { ...
'' ...    
'    for (c_c = 0; c_c < nc_c; c_c += 4)' ...
'    {' ...
'#ifdef IACA_OPERATE' ...
'        IACA_START' ...
'#endif'};
    write_lines_to_file (c_fid, lines);
    
    % Write code to load one column of the right matrix into AVX vectors
    for row = 0:dot_product_length-1
        fprintf (c_fid, '        right_r%u_r_i = _mm256_load_ps (&(B[%u] + c_c)->real);\n', row, row);
        fprintf (c_fid, '        right_r%u_i_r = _mm256_permute_ps (right_r%u_r_i, SWAP_REAL_IMAG_PERMUTE);\n', row, row);
    end
    
    for row = 0:nr_c-1
        [lines, num_vfops] = generate_dot_product_calc (row, dot_product_length, num_vfops);
        write_lines_to_file (c_fid, lines);
    end

    % Complete C function
    lines = {...
'    }' ...
'#ifdef IACA_OPERATE' ...
'    IACA_END' ...
'#endif' ...
'}' ...
''};
    write_lines_to_file (c_fid, lines);
end

function [lines, num_vfops] = generate_dot_product_calc (left_row, dot_product_length, num_vfops)
    dot_product_indices = 0:dot_product_length-1;
    indices = split_dot_product_indices (dot_product_indices);
    lines = {
'' ...
sprintf('        _mm256_store_ps (&(C[%u] + c_c)->real,', left_row)};
    expression_depth = 0;
    num_add_open_brackets = 0;
    [lines, num_vfops] = generate_dot_product_lines (lines, num_vfops, indices, expression_depth, num_add_open_brackets, left_row);
    
    for line_index = 1:length(lines)
        if ~isempty(lines{line_index}) && (lines{line_index}(end) == ')')
            if line_index == length(lines)
                lines{line_index} = [lines{line_index} ');'];
            else
                lines{line_index} = [lines{line_index} ','];
            end
        end
    end
end

function indices = split_dot_product_indices (dot_product_indices)
    if length (dot_product_indices) > 2
        num_left = ceil (length (dot_product_indices) / 2);
        left_dot_product_indices = dot_product_indices(1:num_left);
        right_dot_product_indices = dot_product_indices(num_left+1:end);
        indices = {split_dot_product_indices(left_dot_product_indices) ...
                   split_dot_product_indices(right_dot_product_indices)};
    elseif length (dot_product_indices) == 2
        indices = {dot_product_indices(1) dot_product_indices(2)};
    else
        indices = {dot_product_indices(1)};
    end
end

function [lines, num_vfops] = generate_dot_product_lines (lines, num_vfops, indices, expression_depth, num_add_open_brackets, left_row)
    if ~iscell(indices) || (length(indices) == 2)
        expression_depth = expression_depth + 1;
    end
    indent = (expression_depth + 2) * 4;
    if iscell (indices)
        opened_add = false;
        if length(indices) == 2
            line = [repmat(' ',[1 indent]) '_mm256_add_ps('];
            num_vfops = num_vfops + 1;
            lines = [lines line];
            num_add_open_brackets = num_add_open_brackets + 1;
            opened_add = true;
        end
        [left_lines, num_vfops] = generate_dot_product_lines({}, num_vfops, indices{1}, expression_depth, num_add_open_brackets, left_row);
        for line_index = 1:length(left_lines)
            lines = [lines left_lines{line_index}];
        end
        if length(indices) == 2
            [right_lines, num_vfops] = generate_dot_product_lines({}, num_vfops, indices{2}, expression_depth, num_add_open_brackets, left_row);
            for line_index = 1:length(right_lines)
                lines = [lines right_lines{line_index}];
            end
        end
        
        if opened_add
            lines{end} = [lines{end} ')'];
        end
    else
        mul_prefix = [repmat(' ',[1 indent]) '_mm256_addsub_ps ('];
        lines = [lines [mul_prefix sprintf('_mm256_mul_ps (right_r%u_r_i, left_r%u_c%u_r),', indices, left_row, indices)]];
        lines = [lines [repmat(' ',[1 length(mul_prefix)]) sprintf('_mm256_mul_ps (right_r%u_i_r, left_r%u_c%u_i))', indices, left_row, indices)]];
        num_vfops = num_vfops + 3;
    end
end

function write_lines_to_file (fid, lines)
    for line_num = 1:length(lines)
        fprintf (fid, '%s\n', lines{line_num});
    end
end