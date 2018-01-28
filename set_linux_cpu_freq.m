function set_linux_cpu_freq( min_freq_KHz, max_freq_KHz )
%Set the frequencies for all CPUs under Linux. Uses sudo to gain permission
%to set the CPU frequency, i.e. may prompt for password.
%
%   With no arguments sets ondemand across the maximum supported frequency
%   range.
%
%   With one argument sets a single frequency, using the userspace
%   governor.
%
%   With two arguments sets the min and max frequencies, using the ondemand
%   governor.

    cpu_number = 0;
    while true
        cpu_dir = sprintf ('/sys/devices/system/cpu/cpu%d/cpufreq/', cpu_number);
        freq_fd = fopen ([cpu_dir 'scaling_available_frequencies'], 'r');
        if (freq_fd == -1)
            break
        end
        
        available_freqs = fscanf (freq_fd, '%d');
        fclose (freq_fd);
        
        cpu_number = cpu_number + 1;
        
        if nargin == 0
            % With no frequency specified on the command line, revert to
            % maximum limits
            set_cpu_freq_limits (min(available_freqs), max(available_freqs));
        elseif nargin >= 2
            % Set the input frequency range
            set_cpu_freq_limits (select_closest_frequency (min_freq_KHz), select_closest_frequency (max_freq_KHz));
        else
            % Set the single input frequency
            freq_KHz = select_closest_frequency (min_freq_KHz);
            set_cpu_freq_limits (freq_KHz, freq_KHz);
        end
        
        % Readback to check set the expected value
        fprintf ('CPU %u using governor %s set to frequency range %d - %d KHz, currently %d KHz\n', ...
                 cpu_number, get_current_governor, get_current_min, get_current_max, get_current);
    end

    if cpu_number == 0
        fprintf ('Error: Could not find any CPUFreq controlled CPU cores to manage\n');
        fprintf ('Try adding intel_pstate=disable to the Linux command line\n');
    end

    % Set the min and max frequency scaling limits for the current CPU
    % being processed
    function set_cpu_freq_limits (min_freq, max_freq)
        current_min = get_current_min;
        current_max = get_current_max;
        current_speed = get_current;
        current_governor = get_current_governor;
        
        if min_freq == max_freq
            % Since min and max frequency are the same, select the
            % userspace governor to give a fixed frequency
            governor = 'userspace';
            if (current_speed == min_freq) && (strcmp (current_governor, governor))
                % No change required
                return
            end
        else
            % Select an ondemand governor with the requested frequecy range
            governor = 'ondemand';
            if (current_min == min_freq) && (current_max == max_freq) && (strcmp (current_governor, governor))
                % No change required
                return
            end
        end
        
        system (['echo ' governor '|sudo tee ' cpu_dir 'scaling_governor > /dev/null']);
        
        if strcmp (governor, 'userland')
            system (['echo ' sprintf('%d', min_freq) '|sudo tee ' cpu_dir 'scaling_setspeed > /dev/null']);
        end

        % Select the order in which change min and max to avoid getting an
        % error
        if min_freq > current_max
            system (['echo ' sprintf('%d', max_freq) '|sudo tee ' cpu_dir 'scaling_max_freq > /dev/null']);
            system (['echo ' sprintf('%d', min_freq) '|sudo tee ' cpu_dir 'scaling_min_freq > /dev/null']);
        else
            system (['echo ' sprintf('%d', min_freq) '|sudo tee ' cpu_dir 'scaling_min_freq > /dev/null']);
            system (['echo ' sprintf('%d', max_freq) '|sudo tee ' cpu_dir 'scaling_max_freq > /dev/null']);
        end
    end

    function closest_freq = select_closest_frequency (freq_KHz)
        if (isempty (find(available_freqs == freq_KHz, 1)))
            % The input frequency is not directly supported, select the
            % closest
            tmp = abs(available_freqs - freq_KHz);
            [~, idx] = min(tmp);
            closest_freq = available_freqs(idx);
            fprintf ('Selecting frequency %d as the closest to the supported frequencies on CPU %d\n', ...
                closest_freq, cpu_number);
        else
            closest_freq = freq_KHz;
        end
    end

    function current_governor = get_current_governor
        freq_fd = fopen ([cpu_dir 'scaling_governor']);
        current_governor = fscanf (freq_fd, '%s');
        fclose (freq_fd);
    end

    function current_min = get_current_min
        freq_fd = fopen ([cpu_dir 'scaling_min_freq']);
        current_min = fscanf (freq_fd, '%d');
        fclose (freq_fd);
    end

    function current_max = get_current_max
        freq_fd = fopen ([cpu_dir 'scaling_max_freq']);
        current_max = fscanf (freq_fd, '%d');
        fclose (freq_fd);
    end

    function current = get_current
        freq_fd = fopen ([cpu_dir 'scaling_cur_freq']);
        current = fscanf (freq_fd, '%d');
        fclose (freq_fd);
    end
end
