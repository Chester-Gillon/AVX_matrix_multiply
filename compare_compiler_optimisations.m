function compare_compiler_optimisations
% Compare the optimisations for different gcc compiler versions in terms of
% the operate_block_throughput reported by the IACA analysis for different
% functions.

    gcc4_8_5_results = readtable ('matrix_multiply_attributes_gcc4-8-5.csv','ReadRowNames',true,'Delimiter','comma');
    gcc4_9_3_results = readtable ('matrix_multiply_attributes_gcc4-9-3.csv','ReadRowNames',true,'Delimiter','comma');
    gcc6_3_1_results = readtable ('matrix_multiply_attributes_gcc6-3-1.csv','ReadRowNames',true,'Delimiter','comma');

    compare_compiler_versions(gcc4_8_5_results,gcc4_9_3_results,'4.8.5 to 4.9.3');
    compare_compiler_versions(gcc4_9_3_results,gcc6_3_1_results,'4.9.3 to 6.3.1');
    compare_compiler_versions(gcc4_8_5_results,gcc6_3_1_results,'4.8.5 to 6.3.1');
end

function compare_compiler_versions(results_a,results_b,summary_name)
    multiply_types = {'cmat_mulx_avx' 'cmat_mulx_avx_accumulate' 'cmat_mulx_avx_fma_accumulate'};
    results_percentage = 100 * ((results_a{:,'operate_block_throughput'} ./ results_b{:,'operate_block_throughput'}) - 1);
    for multiply_type_index = 1:length(multiply_types)
        multiply_type = multiply_types{multiply_type_index};
        row_indices = find(strcmp(results_a{:,'multiply_type'},multiply_type));
        figure;
        x = results_a{row_indices,'nr_c'};
        y = results_a{row_indices,'dot_product_length'};
        z = results_percentage(row_indices);
        scatter3(x,y,z,'.');
        xlabel('nr_c','Interpreter','none');
        ylabel('dot_product_length','Interpreter','none');
        zlabel([summary_name ' speedup %'],'Interpreter','none');
        plot_title = {[summary_name ' ' multiply_type]  ...
            sprintf('Min %.2f Max %.2f Mean %.2f', min(z), max(z), mean(z))};
        title(plot_title,'Interpreter','none');
        [~,min_index] = min(z);
        [~,max_index] = max(z);
        fprintf ('%s %s Min at nr_c=%u,dot_product_length=%u Max at %u,%u\n', ...
            plot_title{1}, plot_title{2}, ...
            x(min_index), y(min_index), x(max_index), y(max_index));
    end
end

