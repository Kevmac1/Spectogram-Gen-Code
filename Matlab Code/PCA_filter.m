function Y = PCA_filter(x, ts, indx)

PCA_filter_width = 1; % in seconds, 1 second as suggested in reference
[N_time] = size(x,1);
window_length = ceil(PCA_filter_width/ts); %
% window_length = N_time;
total_number_windows = floor(N_time/window_length);
Y = [];
for i = 1 : total_number_windows
    H = x((i-1)*window_length+1:i*window_length,:);
    H2 = H;
    [V, D] = eig(H2'*H2);
    V = fliplr(V);
    DD(:,i) = diag(D);
    Y = [Y ; H2*V(:,indx)];
end