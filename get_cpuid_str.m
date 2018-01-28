function cpuid_str = get_cpuid_str
% Build a string which contains the CPU model and frequency for using in
% reporting for timing results. The frequency is sampled across all entries
% in /proc/cpuinfo. If a fixed frequency is in use that is reported. If CPU
% frequency scaling is in use the current min and max frequencies are
% reported, as a sign the actual CPU frequency during the tests could
% change.

    cpuid_str = '';
    [status, cpuinfo] = system ('cat /proc/cpuinfo');
    if status == 0
        scanned = textscan (cpuinfo,sprintf('%%s%%s\n'),'Delimiter',':');
        if ~isempty(scanned{1})
            field_names = strtrim(scanned{1,1});
            field_values = strtrim(scanned{1,2});
            model_name_idx = find (strcmp(field_names,'model name'),1);
            if ~isempty(model_name_idx)
                % Find the string before CPU in the model name, which is
                % assumed to the the CPU name
               model_fields = textscan(field_values{model_name_idx},'%s');
               if ~isempty(model_fields{1})
                   cpu_idx = find (strcmp(model_fields{1},'CPU'));
                   if ~isempty(cpu_idx)
                       cpuid_str = model_fields{1}{cpu_idx-1};
                   end
               end
            end
            
            cpu_freq_idxs = find (strcmp(field_names,'cpu MHz'));
            if ~isempty(cpu_freq_idxs)
                if ~isempty(cpuid_str)
                    cpuid_str = [cpuid_str '_'];
                end
                cpu_freqs = uint32(str2double(field_values(cpu_freq_idxs)));
                min_cpu_freq = min(cpu_freqs);
                max_cpu_freq = max(cpu_freqs);
                if (min_cpu_freq == max_cpu_freq)
                    cpuid_str = sprintf ('%s%uMHz', cpuid_str, min_cpu_freq);
                else
                    cpuid_str = sprintf ('%s%u-%uMHz', cpuid_str, min_cpu_freq, max_cpu_freq);
                end
            end
        end
    end

    if isempty(cpuid_str)
        cpuid_str = 'unknown';
    end
end
