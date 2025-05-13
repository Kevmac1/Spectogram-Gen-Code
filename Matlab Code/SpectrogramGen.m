%Spectrogram
close all;
fs = 500; %sampling frequency
tmax = 0;
tmin = 18;
[time, freq, spectrogram]=gaussspec(csi_data,fs,tmax, tmin);
fc = 5.18e9; %carrier frequency
lambda = 3e8/(2*fc); %wavelength
%plot entire spectrogram
imagesc(time, freq*lambda, abs(spectrogram));
axis xy; 
xlabel('Time (s)');
ylabel('Velocity(m/s)');
title('Spectrogram');
colormap(jet);
colorbar; 

velocity = freq * lambda;  %convert frequency to velocity

%filter out velocities > 2 m/s
keep_idx = abs(velocity) <= 2;  %keep only velocities within ±2 m/s


keep_time_indx = time <= 18;
time_filtered = time(keep_time_indx);
velocity_filtered = velocity(keep_idx);
spectrogram_filtered = spectrogram(keep_idx,keep_time_indx);


figure
%plot velocity filtered spectrogram
imagesc(time_filtered, velocity_filtered, abs(spectrogram_filtered));
axis xy;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Spectrogram');
colormap(jet);
colorbar;



%%
%Percentile function 
function [percentile] = percent(current, max)
percentile = (current/max)*100;
end

%calculate starting point to eliminate noise near dc
spectrogram_size = size(spectrogram_filtered);
spectrogram_height = spectrogram_size(1);
starting_idx = ceil(spectrogram_height * (0.15)); %ADJUST

%find total energy for each time step
spectrogram = spectrogram_filtered;
max_f = [];
p = [];
for t = 1:length(spectrogram)
    max_f(t) =  sum(abs(spectrogram(starting_idx:end, t)).^2);
end

for t2 = 1:length(spectrogram)
    m = starting_idx;
    energy_current_f = sum(abs(spectrogram(starting_idx:m, t2)).^2);
    percentile = percent(energy_current_f, max_f(t2));
    while percentile < 45 %ADJUST
        m = m+1;
        energy_current_f = sum(abs(spectrogram(starting_idx:m, t2)).^2);
        percentile = percent(energy_current_f, max_f(t2));
    end
        if max_f(t2) > 0.0125 %ADJUST (min energy)
            p(t2) = freq(m);
        else
            p(t2) = 0;
        end

end

p = p.*lambda;
t_axis = time_filtered;

%plot torso speed
figure
plot(t_axis, p, 'LineWidth', 2);
grid on
title('Torso Speed')
xlabel('Time Steps')
ylabel('Velocity (m/s)')
ylim([0,2])

%%
%Stable selection
p_segment = p;  % Velocity (m/s)
time_steps = 1:length(p_segment);  % Time steps (indices)

%parameters for sliding window
window_size = 130;  %window size (number of points) 170 %%ADJUST
step_size = 10;     %step size for sliding window


variances = [];
avg_velocities = [];
section_start_indices = [];

%loop through the data with a sliding window
for start_idx = 1:step_size:(length(p_segment) - window_size)
    end_idx = start_idx + window_size - 1;
    section_data = p_segment(start_idx:end_idx);

    avg_velocity = mean(section_data);
    if avg_velocity < 0.8 %hard code stable section must be above 0.8 m/s
        continue;
    end
    section_variance = var(section_data);
    
    %store the variance, the average velocity, and the start time of the section
    variances = [variances, section_variance];
    avg_velocities = [avg_velocities, avg_velocity];
    section_start_indices = [section_start_indices, start_idx];
end

%find the section with the least variance around its own average velocity
if ~isempty(variances)
    %find the index of the section with the smallest variance
    [~, min_variance_idx] = min(variances);
    
    %extract relevant info
    least_variance_velocity = avg_velocities(min_variance_idx);
    least_variance_start_idx = section_start_indices(min_variance_idx);
    least_variance_end_idx = least_variance_start_idx + window_size - 1;
    least_variance = variances(min_variance_idx);
    
    %display the results
    disp(['Section with least variance starts at index ', num2str(least_variance_start_idx), ...
          ' and ends at index ', num2str(least_variance_end_idx)]);
    disp(['Average velocity of this section: ', num2str(least_variance_velocity, '%.4f'), ...
          ', Variance: ', num2str(least_variance, '%.6f')]);

    %plot the data and highlight the section with the least variance
    figure;
    plot(time_steps, p_segment, 'LineWidth', 2);
    hold on;
    plot(least_variance_start_idx:least_variance_end_idx, ...
         p_segment(least_variance_start_idx:least_variance_end_idx), ...
         'g', 'LineWidth', 4);  %highlight stable section
    title('Torso Speed with Section of Least Variance Highlighted');
    xlabel('Time Step');
    ylabel('Velocity (m/s)');
    legend('Torso Speed', 'Least Variance Window');
    grid on;
else
    disp('No valid sections found.');
end




%%
%Autocoorelation
%noise near DC possibly involved in this measurement --> may need to change
acf_matrix = [];
lags_matrix = [];
time_range = least_variance_start_idx:least_variance_end_idx;
for i = 1:height(spectrogram) %each frequency step
    [acf,lags] = autocorr(spectrogram(i,time_range));
    acf_matrix = [acf_matrix; acf];
    lags_matrix = [lags_matrix; lags];
end
energy = [];
for j = 1:height(spectrogram)
    energy(j) = sum(spectrogram(j,time_range));
end
%sum energy of each frequecy bin then normalize
energy = energy/max(energy);
weighted_sum = 0;
ac_vector = [];
for k = 1:height(spectrogram)
    ac_vector = [ac_vector; sum(acf_matrix(k,2:3)) * energy(k)];
    weighted_sum = weighted_sum + sum(acf_matrix(k,2:3)) * energy(k);
end
auto_corr = weighted_sum;
%%
%Histogram of AC gradient
grad_auto_corr = gradient(ac_vector);
figure;
num_bins_ac = round(sqrt(length(grad_auto_corr)));
histogram(grad_auto_corr, num_bins_ac); 
xlabel('Gradient Values');
ylabel('Frequency');
title('Histogram of Gradient of Autocorrelation Vector');
grid on;

%mean standard deviation 

mu_ACgrad = mean(grad_auto_corr);          %mean
sigma_ACgrad = std(grad_auto_corr);        %standard deviation


%%
%Avg torso speed
torso_speed_avg = mean(p(time_range));

%%
%Histogram of torso speed gradient
grad_torso_speed = gradient(p(time_range)); 
figure;
num_bins_ts = round(sqrt(length(grad_torso_speed)));
histogram(grad_torso_speed, num_bins_ts); 
xlabel('Gradient Values');
ylabel('Frequency');
title('Histogram of Gradient of Torso Speed Vector');
grid on;

%mean standard deviation 
mu_torso_speed = mean(grad_torso_speed);          %mean
sigma_torso_speed = std(grad_torso_speed);        %standard deviation


%%
%Gait cycle detection

time_segment = time_filtered(time_range);
p_segment = p(time_range);

%peak detection parameters
min_peak_prominence = 0.1 * (max(p_segment) - min(p_segment));
min_peak_distance = 0.3 / mean(diff(time_segment)); % 0.5 seconds

[pks, locs] = findpeaks(p_segment, ...
    'MinPeakProminence', min_peak_prominence, ...
    'MinPeakDistance', min_peak_distance);

if length(locs) >= 2
    peak_times = time_segment(locs);
    gait_cycles = diff(peak_times);
    avg_gait_cycle = mean(gait_cycles);
    stride_length = torso_speed_avg * avg_gait_cycle;
else
    avg_gait_cycle = NaN;
    stride_length = NaN;
    disp('Insufficient peaks detected for gait cycle calculation');
end

%plot peaks on torso speed
figure
plot(time_segment, p_segment, 'LineWidth', 2);
hold on;
plot(peak_times, pks, 'ro', 'MarkerSize', 10);
title('Torso Speed with Gait Cycle Peaks');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('Torso Speed', 'Peaks');
grid on;

%display results
disp('Auto Correlation: ');
disp(auto_corr);
disp('Average Torso Speed (m/s): ');
disp(torso_speed_avg);
disp('Average Gait Cycle Length (s): ');
disp(avg_gait_cycle);
disp('Stride Length (m): ');
disp(stride_length);
disp('Histogram of Torso Speed Gradient Mean: ')
disp(mu_torso_speed);
disp('Histogram of Torso Speed Gradient Std: ')
disp(sigma_torso_speed);
disp('Histogram of AC Gradient Mean: ')
disp(mu_ACgrad);
disp('Histogram of AC Gradient Std: ')
disp(sigma_ACgrad);


feature_vector = [auto_corr; torso_speed_avg; avg_gait_cycle; stride_length; mu_torso_speed; sigma_torso_speed; mu_ACgrad; sigma_ACgrad];
disp(feature_vector)
%%
%NORMALIZED frequency distribution + plotting
% Compute frequency distribution
for j2 = starting_idx:height(spectrogram)
    avg_time = mean(spectrogram(j2,:));
    fd(j2) = avg_time;
end

% Use a fixed bin width approach (assuming uniform spacing)
df = freq(2) - freq(1); % Frequency bin width in Hz
dv = df * lambda / 2;   % Corresponding velocity bin width
% Normalize frequency distribution
total_area = sum(fd) * df; % Compute total area
fd_normalized = fd / total_area; % Normalize fd
% Plot normalized velocity distribution
figure
plot(velocity_filtered, fd_normalized, 'LineWidth', 2);
grid on
title('Normalized Velocity Distribution')
xlabel('Velocity (m/s)')
ylabel('Probability Density')

%%
