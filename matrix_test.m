% Overall test wrapper to generate some random values, and compare the
% matrix multiply results generated by MATLAB against C implementations.
function matrix_test
% Determine cache sizes, for reporting when a data set is expected to fit
% in a cache
[~,result] = system ('getconf LEVEL1_DCACHE_SIZE');
L1_cache_size = str2double(result);
[~,result] = system ('getconf LEVEL2_CACHE_SIZE');
L2_cache_size = str2double(result);
[~,result] = system ('getconf LEVEL3_CACHE_SIZE');
L3_cache_size = str2double(result);

rng('default');
csv_file = fopen ('errors.csv','w');
fprintf (csv_file, 'Function,nr_c,dot_product_length,Num Samples,Max ABS difference,Min duration us,Max duration us,Average duration us,Data set fits in cache\n');
num_timed_iterations = 100;
for nr_c = 2:20
    for dot_product_length = 2:20
        num_samples = 5;
        while num_samples <= 128 * 1024
            weights = complex(rand([nr_c dot_product_length],'single') - 0.5, rand([nr_c dot_product_length],'single') - 0.5);
            samples = complex(rand([dot_product_length num_samples],'single') - 0.5, rand([dot_product_length num_samples],'single') - 0.5);
            matlab_output = weights * samples;
            
            data_set_size_bytes = (nr_c + dot_product_length) * num_samples * 8;
            if data_set_size_bytes < L1_cache_size
                cache_fit = 'L1';
            elseif data_set_size_bytes < L2_cache_size
                cache_fit = 'L2';
            elseif data_set_size_bytes < L3_cache_size
                cache_fit = 'L3';
            else
                cache_fit = 'None';
            end
            
            funcs = {@c_matrix_multiply ...
                @c_avx_split_matrix_multiply @c_avx_interleave_matrix_multiply  ...
                @c_opensal_matrix_multiply ...
                @c_avx_fixed_dimension_matrix_multiply ...
                @c_avx_fixed_dimension_accumulate_matrix_multiply};
            for index=1:length(funcs)
                matrix_func = funcs{index};
                f = functions(matrix_func);
                fprintf ('Calling %s with nr_c=%u dot_product_length=%u num_samples=%d\n', ...
                    f.function, nr_c, dot_product_length, num_samples);
                [c_output, timing] = matrix_func(weights, samples, num_timed_iterations);
                if ~isempty (c_output)
                    differences = c_output - matlab_output;
                    [differences_rows, differences_row_indices] = max(abs(differences));
                    [max_difference, max_difference_col] = max(differences_rows);
                    max_difference_row = differences_row_indices(max_difference_col);
                    fprintf ('Max abs difference = %.8g, at row %d col %d\n', ...
                        max_difference, max_difference_row, max_difference_col);
                    fprintf ('For max difference : matlab_output = %.8g%+.8gi, c_output = %.8g%+.8gi\n', ...
                        real(matlab_output(max_difference_row, max_difference_col)), ...
                        imag(matlab_output(max_difference_row, max_difference_col)), ...
                        real(c_output(max_difference_row, max_difference_col)), ...
                        imag(c_output(max_difference_row, max_difference_col)));
                    fprintf (csv_file,'%s,%u,%u,%d,%.8g,%.1f,%.1f,%.1f,%s\n', f.function, ...
                        nr_c, dot_product_length, num_samples, max_difference, ...
                        timing.min_duration_us, timing.max_duration_us, timing.average_duration_us, cache_fit);
                end
            end
            
            num_samples = num_samples * 3;
        end
    end
end
fclose (csv_file);
end

