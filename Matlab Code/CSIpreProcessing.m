% pre processing code 3D matrix
%BEFORE RUNNING: picoscene .csi file and rename workspace variable Test1
A = Test1;
csi_data = [];
min_length = length(A{1,1}.CSI.CSI);
for i = 1:length(A)
    current_data = A{i,1}.CSI.CSI;
    if (length(current_data) < min_length)
        min_length = length(current_data);
    end
    disp(i);
end

for i = 1:length(A)
    current_data = A{i,1}.CSI.CSI;
    column = current_data(1:min_length,1,1);
    column = [column; current_data(1:min_length,1,2)];
    csi_data = [csi_data, column];
    disp(i);
end

csi_data = csi_data.';

%%
% pre processing code 2D matrix
% csi_data = walktrial2{1,1}.CSI.Mag;
% csi_data = csi_data(:,1:90);
