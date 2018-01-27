function compare_multiply_types
% Compare the performance of different multiply types in terms of the
% operate_block_throughput reported by the IACA analysis.

    gcc4_8_5_results = readtable ('matrix_multiply_attributes_gcc4-8-5.csv','ReadRowNames',true,'Delimiter','comma');
    gcc4_9_3_results = readtable ('matrix_multiply_attributes_gcc4-9-3.csv','ReadRowNames',true,'Delimiter','comma');
    gcc6_3_1_results = readtable ('matrix_multiply_attributes_gcc6-3-1.csv','ReadRowNames',true,'Delimiter','comma');
    
    results = {gcc4_8_5_results, gcc4_9_3_results, gcc6_3_1_results};
    gcc_versions = {'4.8.5', '4.9.3', '6.3.1'};

    compare_multiplies (results,'cmat_mulx_avx_accumulate','cmat_mulx_avx_fma_accumulate',gcc_versions);
    compare_multiplies (results,'cmat_mulx_avx','cmat_mulx_avx_accumulate',gcc_versions);
    compare_multiplies (results,'cmat_mulx_avx','cmat_mulx_avx_fma_accumulate',gcc_versions);
end

function compare_multiplies (results_by_gcc_version,multiply_type_a,multiply_type_b,gcc_versions)
    for version_index = 1:length(gcc_versions)
        results = results_by_gcc_version{version_index};
        multiply_type_a_rows = find(strcmp(results{:,'multiply_type'},multiply_type_a));
        multiply_type_b_rows = find(strcmp(results{:,'multiply_type'},multiply_type_b));
        results_percentage = 100 * ((results{multiply_type_a_rows,'operate_block_throughput'} ./ ...
                                     results{multiply_type_b_rows,'operate_block_throughput'}) - 1);
        figure;
        x = results{multiply_type_a_rows,'nr_c'};
        y = results{multiply_type_a_rows,'dot_product_length'};
        z = results_percentage;
        scatter3(x,y,z,'.');
        xlabel('nr_c','Interpreter','none');
        ylabel('dot_product_length','Interpreter','none');
        zlabel('speedup %','Interpreter','none');
        plot_title = {[gcc_versions{version_index} ' ' multiply_type_a ' -> ' multiply_type_b]  ...
            sprintf('Min %.2f Max %.2f Mean %.2f', min(z), max(z), mean(z))};
        title(plot_title,'Interpreter','none');
        [~,min_index] = min(z);
        [~,max_index] = max(z);
        fprintf ('%s %s Min at nr_c=%u,dot_product_length=%u Max at %u,%u\n', ...
            plot_title{1}, plot_title{2}, ...
            x(min_index), y(min_index), x(max_index), y(max_index));
    end
end