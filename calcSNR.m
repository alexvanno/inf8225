clear; clc;
%% RESULTS_ARCH
cd results_arch
fl = dir; fl = {fl(3:end).name};
inputs = {}; outputs = {};
for i = 1:length(fl)
    if ~isempty(regexp(fl{i},'output'))
        outputs{end+1} = fl{i};
    elseif ~isempty(regexp(fl{i},'input')) && isempty(regexp(fl{i},'Bruit'))
        inputs{end+1} = fl{i};
    end
end
[E_ARCH, R_ARCH] = calcMetrics(inputs, outputs);
cd ..
%% RESULTS_OPTIM_LR
cd results_optim_lr
fl = dir; fl = {fl(3:end).name};
inputs = {}; outputs = {};
for i = 1:length(fl)
    if ~isempty(regexp(fl{i},'output'))
        outputs{end+1} = fl{i};
    elseif ~isempty(regexp(fl{i},'input')) && isempty(regexp(fl{i},'Bruit'))
        inputs{end+1} = fl{i};
    end
end
[E_LR, R_LR] = calcMetrics(inputs, outputs);
cd ..
%% RESULTS_BATCH_SIZE
cd results_batch_size
fl = dir; fl = {fl(3:end).name};
inputs = {}; outputs = {};
for i = 1:length(fl)
    if ~isempty(regexp(fl{i},'output'))
        outputs{end+1} = fl{i};
    elseif ~isempty(regexp(fl{i},'input')) && isempty(regexp(fl{i},'Bruit'))
        inputs{end+1} = fl{i};
    end
end
[E_BS, R_BS] = calcMetrics(inputs, outputs);
cd ..
%% RESULTS_EPOCH
cd results_epoch
fl = dir; fl = {fl(3:end).name};
inputs = {}; outputs = {};
for i = 1:length(fl)
    if ~isempty(regexp(fl{i},'output'))
        outputs{end+1} = fl{i};
    elseif ~isempty(regexp(fl{i},'input')) && isempty(regexp(fl{i},'Bruit'))
        inputs{end+1} = fl{i};
    end
end
[E_EPOCH, R_EPOCH] = calcMetrics(inputs, outputs);
cd ..

% clear; clc;
% archList4 = {'arch1', 'arch2', 'arch3', 'arch4'};
% for i = 1:length(archList4)
%     % Get images
%     input = imread(['results/' archList4{i} '_100e_lr1e-4_input.png']);
%     input = im2double(input(:,:,1));
%     inputNoise = imread(['results/' archList4{i} '_100e_lr1e-4_inputBruit.png']);
%     inputNoise = im2double(inputNoise(:,:,1));
%     output = imread(['results/' archList4{i} '_100e_lr1e-4_output.png']);
%     output = im2double(output(:,:,1));
%     
%     difNoise = abs(inputNoise - input);
%     difOutput = abs(output - input);
%     
%     avgNoise(i) = sum(sum(difNoise))/50;
%     avgOutput(i) = sum(sum(difOutput))/50;
%     
%     stdNoise(i) = std(difNoise(input==0));
%     stdOutput(i) = std(difOutput(input==0));
%     
%     rNoise(i) = corr2(input, inputNoise);
%     rOutput(i) = corr2(input, output);
%     
%     % http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154160
% end
% 
% archList5 = {'arch3'};
% for i = 1:length(archList5)
%     % Get images
%     input = imread(['results/' archList5{i} '_100e_lr1e-5_input.png']);
%     input = im2double(input(:,:,1));
%     inputNoise = imread(['results/' archList5{i} '_100e_lr1e-5_inputBruit.png']);
%     inputNoise = im2double(inputNoise(:,:,1));
%     output = imread(['results/' archList5{i} '_100e_lr1e-5_output.png']);
%     output = im2double(output(:,:,1));
%     
%     difNoise = abs(inputNoise - input);
%     difOutput = abs(output - input);
%     
%     avgNoise(end+1) = sum(sum(difNoise))/50;
%     avgOutput(end+1) = sum(sum(difOutput))/50;
%     
%     stdNoise(end+1) = std(difNoise(input==0));
%     stdOutput(end+1) = std(difOutput(input==0));
%     
%     rNoise(end+1) = corr2(input, inputNoise);
%     rOutput(end+1) = corr2(input, output);
% end