close all;
%Plot the spectrogram
fs = 500;
tmax = 0;
tmin = 50;
[time, freq, spectrogram]=gaussspec(csi_data,fs,tmax, tmin);
fc = 5.18e9; %carrier frequency
lambda = 3e8/(2*fc); %wavelength
imagesc(time, freq*lambda, abs(spectrogram));
axis xy;  % Flip y-axis so that frequency 0 is at the bottom
xlabel('Time (s)');
ylabel('Velocity(m/s)');
title('Spectrogram');
colormap(jet);
colorbar; 

%freq*lambda in spectrogram if you want to plot versus velocity

%%
%Percentile function 
% function [percentile] = percent(current, max)
% percentile = (current/max)*100;
% end

%find total energy for each time step
max_f = [];
p = [];
for t = 1:2351
    max_f(t) =  sum(abs(spectrogram(:, t)).^2);
end

for t2 = 1:2351
    m = 1;
    energy_current_f = sum(abs(spectrogram(1:m, t2)).^2);
    percentile = percent(energy_current_f, max_f(t2));
    while percentile < 50
        m = m+1;
        energy_current_f = sum(abs(spectrogram(1:m, t2)).^2);
        percentile = percent(energy_current_f, max_f(t2));
    end
        p(t2) = freq(m);

end

p = p.*lambda;

%Plot torso speed
figure
plot(1:length(p), p, 'LineWidth', 2);
grid on
title('Torso Speed')
xlabel('Time Steps')
ylabel('Velocity (m/s)')

%%
%Gait cycle detection and stable period location
p_segment = p;
time_segment = time;

% Peak detection parameters
min_peak_prominence = 0.1 * (max(p_segment) - min(p_segment));
min_peak_distance = 0.5 / mean(diff(time_segment)); % 0.5 seconds

% Detect peaks (local maxima)
[pks, locs] = findpeaks(p_segment, ...
    'MinPeakProminence', min_peak_prominence, ...
    'MinPeakDistance', min_peak_distance);

% Detect minima (local minima) by finding peaks on the inverted signal
[min_pks, min_locs] = findpeaks(-p_segment, ...
    'MinPeakProminence', min_peak_prominence, ...
    'MinPeakDistance', min_peak_distance);

% Plot torso speed with detected peaks and minima
figure
plot(time, p, 'LineWidth', 2);
hold on;

% Plot peaks (local maxima)
plot(time_segment(locs), pks, 'ro', 'MarkerSize', 10);

% Plot minima (local minima)
plot(time_segment(min_locs), -min_pks, 'bo', 'MarkerSize', 10);

title('Torso Speed with Gait Cycle Peaks and Minima');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('Torso Speed', 'Peaks', 'Minima');
grid on;


p_segment = p;  % Torso speed data (velocity in m/s)
time_steps = 1:length(p_segment);  % Time steps (indices) as time points

% Parameters for sliding window
window_size = 500;  % Window size (number of points)
step_size = 10;  % Step size for sliding window

% Target velocity (1 m/s)
target_velocity = 1;

% Threshold for how close we want the average velocity to be to 1 m/s
velocity_tolerance = 0.1; % Can be adjusted depending on the desired range

% Initialize variables to store variances, average velocities, and section start times
variances = [];
avg_velocities = [];
section_start_indices = [];

% Loop through the data with a sliding window
for start_idx = 1:step_size:length(p_segment) - window_size
    end_idx = start_idx + window_size - 1;
    section_data = p_segment(start_idx:end_idx);
    avg_velocity = mean(section_data);
    
    % Check if the average velocity is close to the target velocity (1 m/s)
    if abs(avg_velocity - target_velocity) <= velocity_tolerance
        section_variance = var(section_data);
        % Store the variance, the average velocity, and the start time of the section
        variances = [variances, section_variance];
        avg_velocities = [avg_velocities, avg_velocity];
        section_start_indices = [section_start_indices, start_idx];  % Store start index instead of time
    end
end

% Find the section with the least variance around the target velocity
if ~isempty(variances)
    % Find the index of the section with the smallest variance
    [~, min_variance_idx] = min(variances);
    
    % Find the average velocity of this section and its start and end indices
    least_variance_velocity = avg_velocities(min_variance_idx);
    least_variance_start_idx = section_start_indices(min_variance_idx);
    least_variance_end_idx = least_variance_start_idx + window_size - 1;
    least_variance = variances(min_variance_idx);
    
    % Display the results
    disp(['Section with least variance around 1 m/s starts at time step ', num2str(least_variance_start_idx), ' and ends at time step ', num2str(least_variance_end_idx)]);
  
    % Plot the data and highlight the section with the least variance
    figure;
    plot(time_steps, p_segment, 'LineWidth', 2);  % Plot using time steps (indices)
    hold on;
    plot([least_variance_start_idx, least_variance_end_idx], [min(p_segment), min(p_segment)], 'g', 'LineWidth', 4); % Highlight section with least variance
    title('Torso Speed with Section of Least Variance Around 1 m/s Highlighted');
    xlabel('Time Step');  % X-axis is now in time steps (indices)
    ylabel('Velocity (m/s)');
    legend('Torso Speed', 'Section with Least Variance Around 1 m/s');
    grid on;
else
    disp('No section found with average velocity close to 1 m/s');
end



%%
%Autocoorelation
acf_matrix = [];
lags_matrix = [];
time_range = least_variance_start_idx:least_variance_end_idx;
for i = 1:41 %each time step
    [acf,lags] = autocorr(spectrogram(i,time_range));
    acf_matrix = [acf_matrix; acf];
    lags_matrix = [lags_matrix; lags];
end
energy = [];
for j = 1:41
    energy(j) = sum(spectrogram(j,time_range));
end
%sum energy of each frequecy bin then normalize
energy = energy/max(energy);
weighted_sum = 0;
ac_vector = [];
for k = 1:41
    ac_vector = [ac_vector; sum(acf_matrix(k,2:3)) * energy(k)];
    weighted_sum = weighted_sum + sum(acf_matrix(k,2:3)) * energy(k);
end
auto_corr = weighted_sum;
%%
%histogram of AC gradient
grad_auto_corr = gradient(ac_vector);
figure;
num_bins_ac = round(sqrt(length(grad_auto_corr)));
histogram(grad_auto_corr, num_bins_ac); 
xlabel('Gradient Values');
ylabel('Frequency');
title('Histogram of Gradient of Autocorrelation Vector');
grid on;

%%
%avg torso speed
torso_speed_avg = mean(p(time_range));

%%
%histogram of torso speed gradient
grad_torso_speed = gradient(p(time_range)); 
figure;
num_bins_ts = round(sqrt(length(grad_torso_speed)));
histogram(grad_auto_corr, num_bins_ts); 
xlabel('Gradient Values');
ylabel('Frequency');
title('Histogram of Gradient of Torso Speed Vector');
grid on;
%%
% Combine peaks and minima information
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

%%
% Display results
disp('Auto Correlation: ');
disp(auto_corr);
disp('Average Torso Speed (m/s): ');
disp(torso_speed_avg);
disp('Average Gait Cycle Length (s): ');
disp(avg_gait_cycle);
disp('Stride Length (m): ');
disp(stride_length);

%%
%NORMALIZED frequency distribution + plotting
% Compute frequency distribution
for j2 = 1:41
    avg_time = mean(spectrogram(j2,:));
    fd(j2) = avg_time;
end
% Convert frequency values to velocity using Doppler effect
velocity = freq * lambda;
% Use a fixed bin width approach (assuming uniform spacing)
df = freq(2) - freq(1); % Frequency bin width in Hz
dv = df * lambda / 2;   % Corresponding velocity bin width
% Normalize frequency distribution
total_area = sum(fd) * df; % Compute total area
fd_normalized = fd / total_area; % Normalize fd
% Plot normalized velocity distribution
figure
plot(velocity, fd_normalized, 'LineWidth', 2);
grid on
title('Normalized Velocity Distribution')
xlabel('Velocity (m/s)')
ylabel('Probability Density')

%%
function [percentile] = percent(current, max)
percentile = (current/max)*100;
end