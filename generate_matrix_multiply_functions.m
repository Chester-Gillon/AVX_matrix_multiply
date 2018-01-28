function generate_matrix_multiply_functions
% Generate the C code for the matrix multiply functions, also creating a
% table which lists all the generated functions and the number of vector
% floating point operations.

    mat_mulx_attribs = max_mulx_attribs_table;
    warn_struct = warning ('off', 'MATLAB:table:RowsAddedExistingVars');
    mat_mulx_attribs = generate_avx_fixed_dimension_matrix_multiply (mat_mulx_attribs);
    mat_mulx_attribs = generate_avx_fixed_dimension_accumulate_matrix_multiply (mat_mulx_attribs);
    mat_mulx_attribs = generate_avx_fma_fixed_dimension_accumulate_matrix_multiply (mat_mulx_attribs);
    warning (warn_struct);
    writetable (mat_mulx_attribs, 'matrix_multiply_attributes.csv', 'WriteRowNames', true);
end

% Create an empty table of the maxtrix multiply function attributes
function mat_mulx_attribs = max_mulx_attribs_table
    multiply_type = {};
    nr_c = [];
    dot_product_length = [];
    num_vfops = [];
    mat_mulx_attribs = table (multiply_type, nr_c, dot_product_length, num_vfops, 'RowNames', {});

    mat_mulx_attribs.Properties.VariableDescriptions{'multiply_type'} = ...
        'The type of matrix multiply operation, which is the prefix for the function name';
    mat_mulx_attribs.Properties.VariableDescriptions{'nr_c'} = ...
        'Number of rows for left input matrix and output matrix';
    mat_mulx_attribs.Properties.VariableDescriptions{'dot_product_length'} = ...
        'Number of rows for right input matrix';
    mat_mulx_attribs.Properties.VariableDescriptions{'num_vfops'} = ...
        'Number of 256-bit vector floating point operations performed to produce 4 complex columns of the output matrix';
end
