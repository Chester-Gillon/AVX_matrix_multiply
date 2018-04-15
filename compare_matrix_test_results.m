function compare_matrix_test_results
% Compare the timing results produced by matrix_test for different maxtrix
% multiply functions, in terms of reporting which implementation is the
% fastest based upon the median time.

    [filenames, pathname] = uigetfile('*.csv','Select matrix_test result file(s)','MultiSelect','on');
    if pathname == 0
        return
    end
    
    warn_struct = warning ('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    if iscell(filenames)
        for filename_index = 1:length(filenames)
            result_pathname = fullfile(pathname,filenames{filename_index});
            process_matrix_test_result_file (result_pathname);
        end
    else
        result_pathname = fullfile(pathname,filenames);
        process_matrix_test_result_file (result_pathname);
    end
    warning (warn_struct);
end

function process_matrix_test_result_file (result_pathname)
    multiply_types = {'c_avx_fixed_dimension_matrix_multiply' 'c_avx_fixed_dimension_accumulate_matrix_multiply' 'c_avx_fixed_dimension_fma_accumulate_matrix_multiply'};
    data_sizes = {'L1', 'L2', 'L3', 'None'};
    alloc_methods = {'avx_align', 'l1d_stride'};
    
    % @todo Use detectImportOptions() to set the delimiter, to prevent
    %       rows with "Duration us" values inserting spurious rows. 
    opts = detectImportOptions (result_pathname,'Delimiter',',');
    full_results = readtable (result_pathname,opts);
    fprintf ('\nProcessing %s\n', result_pathname);
    
    for alloc_methods_index = 1:length(alloc_methods)
        alloc_method = alloc_methods{alloc_methods_index};
        
        % Select only results for wanted multiply function types and matrix
        % allocation method
        selected_rows = [];
        for type_index = 1:length(multiply_types)
            selected_rows = [selected_rows; find(and(strcmp(full_results.Function,multiply_types{type_index}), ...
                                                     strcmp(full_results.MatrixAllocationMethod,alloc_method)))];
        end
        results = full_results(selected_rows,:);
        
        % Sort by Samples Per Second so that for each unique combination the
        % fastest function is the first row
        results = sortrows(results,'SamplesPerSecond','descend');
        
        for data_sizes_index = 1:length(data_sizes)
            data_size = data_sizes{data_sizes_index};
            data_size_rows = find(strcmp(results{:,'DataSetFitsInCache'},data_size));
            unique_data_sizes = unique(results{data_size_rows,{'nr_c' 'dot_product_length' 'NumSamples'}},'rows');
            fastest_function_histogram = struct('func_order',{},'count',[]);
            
            % Produce a histogram of the count of the relative performance of
            % the different matrix multiply functions across all data sizes
            % which fit in the same size cache.
            for unique_data_size_index = 1:size(unique_data_sizes,1)
                nr_c = unique_data_sizes(unique_data_size_index,1);
                dot_product_length = unique_data_sizes(unique_data_size_index,2);
                num_samples = unique_data_sizes(unique_data_size_index,3);
                matching_rows = find(and(and(results.nr_c == nr_c,results.dot_product_length == dot_product_length),...
                    results.NumSamples == num_samples));
                func_order = results{matching_rows,'Function'};
                func_index = 1;
                match_found = false;
                while ~match_found && ~isempty(fastest_function_histogram) && (func_index <= length(fastest_function_histogram.func_order))
                    if all(strcmp(func_order,fastest_function_histogram.func_order{func_index}))
                        match_found = true;
                    else
                        func_index = func_index + 1;
                    end
                end
                if match_found
                    fastest_function_histogram.count(func_index) = fastest_function_histogram.count(func_index) + 1;
                else
                    fastest_function_histogram(1).count(func_index) = 1;
                    fastest_function_histogram(1).func_order{func_index} = func_order;
                end
            end
            
            fprintf ('Data set fits in cache: %s  Matrix Allocation Method: %s\n', data_size, alloc_method);
            [~,sorted_indices] = sort (fastest_function_histogram.count,'descend');
            for func_index = sorted_indices
                func_order = fastest_function_histogram.func_order{func_index};
                for name_index = 1:length(func_order)
                    if name_index == 1
                        fprintf (' %s', func_order{name_index});
                    else
                        fprintf (' > %s', func_order{name_index});
                    end
                end
                fprintf (' : %u\n', fastest_function_histogram.count(func_index));
            end
        end
    end
end