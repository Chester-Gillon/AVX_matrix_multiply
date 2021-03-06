% Overall test wrapper to generate some random values, and compare the
% matrix multiply results generated by MATLAB against C implementations.
%
% The following command can be used to allocate sufficent huge pages for
% the test to use a matrix allocation method of l1d_stride
% echo 25 | sudo tee /proc/sys/vm/nr_hugepages
function matrix_test
% Determine cache sizes, for reporting when a data set is expected to fit
% in a cache
[~,result] = system ('getconf LEVEL1_DCACHE_SIZE');
L1_cache_size = str2double(result);
[~,result] = system ('getconf LEVEL2_CACHE_SIZE');
L2_cache_size = str2double(result);
[~,result] = system ('getconf LEVEL3_CACHE_SIZE');
L3_cache_size = str2double(result);

if ~cpu_supports('avx')
    fprintf ('AVX is not supported by CPU, unable to run any tests')
    return
end

% Select AVX functions, and optionally FMA functions if supported by the
% CPU.
funcs = {@c_matrix_multiply ...
    @c_opensal_matrix_multiply ...
    @c_avx_split_matrix_multiply ...
    @c_avx_interleave_matrix_multiply};
if cpu_supports('fma')
    funcs = [funcs ...
        {@c_avx_fma_interleave_matrix_multiply}];
end
funcs = [funcs ...
    {@c_avx_fixed_dimension_matrix_multiply ...
     @c_avx_fixed_dimension_accumulate_matrix_multiply}];
if cpu_supports('fma')
    funcs = [funcs ...
        {@c_avx_fixed_dimension_fma_accumulate_matrix_multiply}];
end

% Build a unique filename for the results
results_filename = [datestr(now,'YYYYmmddTHHMMSS') '_' get_cpuid_str '_matrix_test.csv'];

perf_event_names = { ...
    'hw_instructions', ...
    'hw_ref_cpu_cycles', ...
    'l1d_read_access', ...
    'l1d_read_miss', ...
    'l1d_write_access', ...
    'l1d_write_miss', ...
    'llc_read_access', ...
    'llc_read_miss', ...
    'llc_write_access', ...
    'llc_write_miss'};

rng('default');
csv_file = fopen (results_filename,'w');
fprintf (csv_file, 'Function,nr_c,dot_product_length,Num Samples,Block Other CPUs,Matrix Allocation Method,Max ABS difference,Min duration us,Max duration us,Median duration us,Data set fits in cache,Samples per second,Min Outer RDTSC,Max Outer RDTSC, Median Outer RDTSC,Min Inner RDTSC,Max Inner RDTSC, Median Inner RDTSC,Self Page Reclaims,Self Page Faults,RSS Increase,Self User Time us,Self System Time us,Thread User Time us,Thread System Time us,Count NAN,Count Infinite,Count Zero,Count Subnormal,Count Normal,');
for perf_event_index = 1:length(perf_event_names)
    fprintf (csv_file, '%s,', perf_event_names{perf_event_index});
end
fprintf (csv_file, 'Durations us\n');
num_timed_iterations = 50;
for nr_c = 2:20
    for dot_product_length = 2:20
        for repeat = 1%1:5
            for block_other_cpus = 0
                num_samples = 14;
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
                    
                    for alloc_method_cell = {'avx_align' 'l1d_stride'}
                        alloc_method = alloc_method_cell{1};
                        for index=1:length(funcs)
                            matrix_func = funcs{index};
                            f = functions(matrix_func);
                            fprintf ('Calling %s with nr_c=%u dot_product_length=%u num_samples=%d block_other_cpus=%u alloc_method=%s\n', ...
                                f.function, nr_c, dot_product_length, num_samples, block_other_cpus, alloc_method);
                            [c_output, timing_results] = matrix_func(weights, samples, num_timed_iterations, block_other_cpus, alloc_method);
                            if ~isempty (c_output)
                                start_times_ns = (uint64(timing_results.start_times_tv_sec) .* uint64(1E9)) + (uint64(timing_results.start_times_tv_nsec));
                                stop_times_ns = (uint64(timing_results.stop_times_tv_sec) .* uint64(1E9)) + (uint64(timing_results.stop_times_tv_nsec));
                                durations_us = double(stop_times_ns - start_times_ns) ./ 1000;
                                samples_per_second = num_samples / (median (durations_us) / 1E6);
                                inner_rdtsc_durations = timing_results.stop_times_inner_rdtsc - timing_results.start_times_inner_rdtsc;
                                outer_rdtsc_durations = timing_results.stop_times_outer_rdtsc - timing_results.start_times_outer_rdtsc;
                                differences = c_output - matlab_output;
                                c_classification = fp_classify_matrix (c_output);
                                [differences_rows, differences_row_indices] = max(abs(differences));
                                [max_difference, max_difference_col] = max(differences_rows);
                                max_difference_row = differences_row_indices(max_difference_col);
                                self_page_reclaims = timing_results.stop_self_usage.ru_minflt - timing_results.start_self_usage.ru_minflt;
                                self_page_faults = timing_results.stop_self_usage.ru_majflt - timing_results.stop_self_usage.ru_majflt;
                                rss_increase = timing_results.stop_self_usage.ru_maxrss - timing_results.start_self_usage.ru_maxrss;
                                self_user_time_us = ((timing_results.stop_self_usage.ru_utime_tv_sec * 1E6) + timing_results.stop_self_usage.ru_utime_tv_usec) - ...
                                    ((timing_results.start_self_usage.ru_utime_tv_sec * 1E6) + timing_results.start_self_usage.ru_utime_tv_usec);
                                self_system_time_us = ((timing_results.stop_self_usage.ru_stime_tv_sec * 1E6) + timing_results.stop_self_usage.ru_stime_tv_usec) - ...
                                    ((timing_results.start_self_usage.ru_stime_tv_sec * 1E6) + timing_results.start_self_usage.ru_stime_tv_usec);
                                thread_user_time_us = ((timing_results.stop_thread_usage.ru_utime_tv_sec * 1E6) + timing_results.stop_thread_usage.ru_utime_tv_usec) - ...
                                    ((timing_results.start_thread_usage.ru_utime_tv_sec * 1E6) + timing_results.start_thread_usage.ru_utime_tv_usec);
                                thread_system_time_us = ((timing_results.stop_thread_usage.ru_stime_tv_sec * 1E6) + timing_results.stop_thread_usage.ru_stime_tv_usec) - ...
                                    ((timing_results.start_thread_usage.ru_stime_tv_sec * 1E6) + timing_results.start_thread_usage.ru_stime_tv_usec);
                                fprintf ('Max abs difference = %.8g, at row %d col %d\n', ...
                                    max_difference, max_difference_row, max_difference_col);
                                fprintf ('For max difference : matlab_output = %.8g%+.8gi, c_output = %.8g%+.8gi\n', ...
                                    real(matlab_output(max_difference_row, max_difference_col)), ...
                                    imag(matlab_output(max_difference_row, max_difference_col)), ...
                                    real(c_output(max_difference_row, max_difference_col)), ...
                                    imag(c_output(max_difference_row, max_difference_col)));
                                fprintf (csv_file,'%s,%u,%u,%u,%u,%s,%.8g,%.1f,%.1f,%.1f,%s,%.0f,', f.function, ...
                                    nr_c, dot_product_length, num_samples, block_other_cpus, alloc_method, max_difference, ...
                                    min (durations_us), max (durations_us), median (durations_us), ...
                                    cache_fit, samples_per_second);
                                fprintf (csv_file,'%u,%u,%u,', min(outer_rdtsc_durations), max(outer_rdtsc_durations), median(outer_rdtsc_durations));
                                fprintf (csv_file,'%u,%u,%u,', min(inner_rdtsc_durations), max(inner_rdtsc_durations), median(inner_rdtsc_durations));
                                fprintf (csv_file,'%u,%u,%u,', self_page_reclaims, self_page_faults, rss_increase);
                                fprintf (csv_file,'%u,%u,%u,%u,', self_user_time_us, self_system_time_us, thread_user_time_us, thread_system_time_us);
                                fprintf (csv_file,'%u,%u,%u,%u,%u,', c_classification.count_nan, c_classification.count_infinite, ...
                                    c_classification.count_zero, c_classification.count_subnormal, c_classification.count_normal);
                                for perf_event_index = 1:length(perf_event_names)
                                    if isfield (timing_results.perf_events, perf_event_names{perf_event_index})
                                        fprintf (csv_file, '%u,', timing_results.perf_events.(perf_event_names{perf_event_index}));
                                    else
                                        fprintf (csv_file, ',');
                                    end
                                end
                                if ((max(durations_us) - median(durations_us)) > 1000) && ...
                                        (max(durations_us) > (3 * median (durations_us)))
                                    fprintf (csv_file,'%.1f,',durations_us);
                                end
                                fprintf (csv_file,'\n');
                            end
                        end
                    end
                    
                    num_samples = num_samples * 2;
                end
            end
        end
    end
end
fclose (csv_file);
end

