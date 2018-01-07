function matrix_multiply_iaca_analysis (gcc_executable)
% Perform analysis of the left matrix load time and operate loop time of
% the matrix multiply functions using the Intel Architecture Code Analyzer
% (IACA).
%
% The source of the functions to analyse is the
% matrix_multiple_attributes.csv table created by
% generate_matrix_multiply_functions.
%
% The specific GCC compiler to be used to compile the functions with the
% IACA marks is an input, so can analyse if different GCC versions use more
% agressive optimisation.

    if nargin < 1
        fprintf ('Error: gcc_executable not provided\n');
        return
    end
    
    gcc_version = get_gcc_version (gcc_executable);
    if isempty (gcc_version)
        fprintf ('Error: Failed to obtain GCC version for %s\n', gcc_executable);
        return
    end

    % Read the table of functions to be analysed, and create the
    % directories to contain the IACA analysis results
    mat_mulx_attribs = readtable ('matrix_multiply_attributes.csv', 'ReadRowNames', true, 'Delimiter', 'comma');
    matrix_multiply_types = unique (mat_mulx_attribs{:,'multiply_type'});
    for index = 1:length(matrix_multiply_types)
        [~,~] = mkdir (['mat_mul_iaca_analysis/' matrix_multiply_types{index}]);
    end
    
    % Add columns for IACA analysis summary
    mat_mulx_attribs.load_left_block_throughput = nan(height(mat_mulx_attribs),1);
    mat_mulx_attribs.load_left_throughput_bottlenck = repmat({''},height(mat_mulx_attribs),1);
    mat_mulx_attribs.load_left_num_uops = nan(height(mat_mulx_attribs),1);
    mat_mulx_attribs.operate_block_throughput = nan(height(mat_mulx_attribs),1);
    mat_mulx_attribs.operate_throughput_bottlenck = repmat({''},height(mat_mulx_attribs),1);
    mat_mulx_attribs.operate_num_uops = nan(height(mat_mulx_attribs),1);
    
    for row = 1:height (mat_mulx_attribs)
        compile_matrix_multiply_iaca_wrapper (gcc_executable, row, mat_mulx_attribs, 'IACA_LOAD_LEFT');
        [block_throughput, throughput_bottleneck, num_uops] = run_iaca_analysis (row, mat_mulx_attribs, 'load_left');
        mat_mulx_attribs.load_left_block_throughput(row) = block_throughput;
        mat_mulx_attribs.load_left_throughput_bottlenck{row} = throughput_bottleneck;
        mat_mulx_attribs.load_left_num_uops(row) = num_uops;

        compile_matrix_multiply_iaca_wrapper (gcc_executable, row, mat_mulx_attribs, 'IACA_OPERATE');
        [block_throughput, throughput_bottleneck, num_uops] = run_iaca_analysis (row, mat_mulx_attribs, 'operate');
        mat_mulx_attribs.operate_block_throughput(row) = block_throughput;
        mat_mulx_attribs.operate_throughput_bottlenck{row} = throughput_bottleneck;
        mat_mulx_attribs.operate_num_uops(row) = num_uops;
    end
    
    summary_filename = ['matrix_multiply_attributes_gcc' strrep(gcc_version,'.','-') '.csv'];
    writetable (mat_mulx_attribs, summary_filename, 'WriteRowNames', true);
end

% Obtain the GCC version, by parsing the output of the dumpspecs option
function gcc_version = get_gcc_version (gcc_executable)
    gcc_version = '';
    [status, specs] = system ([gcc_executable ' -dumpspecs']);
    if status == 0
        version_idx = strfind (specs, '*version:');
        if ~isempty (version_idx)
            gcc_version = sscanf (specs(version_idx:end), '*version:\n%s');
        end
    end
end

% Use GCC to compile the IACA wrapper for one matrix multiply function
function compile_matrix_multiply_iaca_wrapper (gcc_executable, row, mat_mulx_attribs, iaca_marker_enable)
    opensal_root = fullfile(getenv('HOME'),'opensal-1.0.0');
    opensal_include = fullfile(opensal_root,'include');
    iaca_root = fullfile(getenv('HOME'),'iaca-lin64');
    gcc_command = [gcc_executable ' -c matrix_multiply_iaca_wrapper.c -o matrix_multiply_iaca_wrapper.o' ...
        ' -O3 -mavx -mfma' ...
        ' -I ' opensal_include ' -I ' iaca_root ...
        ' -DFUNC_FILENAME=\"mat_mul_fragments/' mat_mulx_attribs.multiply_type{row} '/' mat_mulx_attribs.Row{row} '.c\"' ...
        ' -D' iaca_marker_enable];
    [status, error_text] = system (gcc_command);
    if status
        error ('compile of IACA wrapper for %s failed:\n%s\n', mat_mulx_attribs.Row{row}, error_text);
    end
end

% Perform the IACA analysis for one matrix multiply function, for the
% Haswell architecture.
% The block throughput and troughput bottleneck are returned for saving in
% a summary table.
% The IACA throughout analysis report and trace are saved to filenames
% based upon the matrix multiply function name.
function [block_throughput, throughput_bottleneck, num_uops] = run_iaca_analysis (row, mat_mulx_attribs, analysed_section)
    iaca_root = fullfile(getenv('HOME'),'iaca-lin64');
    iaca_executable = fullfile (iaca_root, 'iaca');
    log_directory = ['mat_mul_iaca_analysis/' mat_mulx_attribs.multiply_type{row}];
    log_basename = [log_directory '/' mat_mulx_attribs.Row{row} '_' analysed_section];
    iaca_command = [iaca_executable ' -arch HSW -trace ' log_basename '_trace.txt matrix_multiply_iaca_wrapper.o'];
    [status, iaca_output] = system (iaca_command);
    if status
        error ('IACA analysis for %s failed:\n%s\n', mat_mulx_attribs.Row{row}, iaca_output);
    end
    
    throughput_fid = fopen ([log_basename '_throughput.txt'], 'wt');
    fprintf (throughput_fid, '%s', iaca_output);
    fclose (throughput_fid);
    
    block_throughput = NaN;
    throughput_bottleneck = '';
    num_uops = NaN;
    idx = strfind (iaca_output, 'Block Throughput:');
    if ~isempty (idx)
        block_throughput = sscanf (iaca_output(idx(1):end), 'Block Throughput: %f Cycles');
    end
    idx = strfind (iaca_output, 'Throughput Bottleneck:');
    if ~isempty (idx)
        throughput_bottleneck = sscanf (iaca_output(idx(1):end), sprintf('Throughput Bottleneck: %%[^\n]'));
    end
    idx = strfind (iaca_output, 'Total Num Of Uops:');
    if ~isempty (idx)
        num_uops = sscanf (iaca_output(idx(1):end), 'Total Num Of Uops: %u');
    end
end
