% Author: Ma, yifan (yifan.ma@mailbox.tu-dresden.de)
% Date: 14.05.2021
% The MIT License
% Copyright (c) [2021] Ma yifan


clc
clear 
close all

%% Load Dataset
load Dataset/mixedL.mat

%% Load U-Net Layer
% load Network/Layer_128_64.mat
load Network/UNet_Filter8.mat

%% Variables
layers=size(lbl_Trn,3); % depth layers
Testidx = randperm(size(img_Test,4),5); % The number of expamle images

%% Set training options.
options = trainingOptions('adam', ...   %adam
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',25, ...
    'VerboseFrequency',5,...
    'ValidationFrequency',5,...
    'ValidationData',{img_Vld,lbl_Vld},...
    'ExecutionEnvironment','auto',...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    ... %'CheckpointPath', 'Checkpoints',...
    'MiniBatchSize',32);
%% Training
[net, info] =  trainNetwork(img_Trn,lbl_Trn,lgraph,options);

%% Load Trained Net(For Test)
load TrainedNet/UNetT_mnistL_F16.mat

%% Test Example
Pre=predict(net,img_Test(:,:,layers,Testidx));
Label=lbl_Test(:,:,1,Testidx);
figure
tiledlayout(2,5,'TileSpacing','none')
for i=1:size(Testidx,2)
    nexttile
    imshow(Label(:,:,1,i),[0 255])
end
title('Label')
for j=1:size(Testidx,2)
    nexttile
    imshow(Pre(:,:,1,j),[0 255])
end
title('Prediction')

%% Correlation Coefficient
ccTrn = zeros(1,size(lbl_Trn,4));
ccVld = zeros(1,size(lbl_Vld,4));
ccTest = zeros(1,size(lbl_Test,4));

for i =1:size(lbl_Trn,4)
    rou=corrcoef(predict(net,img_Trn(:,:,1,i)),lbl_Trn(:,:,1,i));
    ccTrn(1,i)=rou(1,2);
end
fprintf('correlation coefficent of Training dataset is:%.4f\n', sum(ccTrn)./size(lbl_Trn,4));

for i =1:size(lbl_Vld,4)
    rou=corrcoef(predict(net,img_Vld(:,:,1,i)),lbl_Vld(:,:,1,i));
    ccVld(1,i)=rou(1,2);
end
fprintf('correlation coefficent of Validation dataset is:%.4f\n', sum(ccVld)./size(lbl_Vld,4));

tic
for i =1:size(img_Test,4)
    rou=corrcoef(predict(net,img_Test(:,:,1,i)),lbl_Test(:,:,1,i));
    ccTest(1,i)=rou(1,2);
end

fprintf('correlation coefficent of Test dataset is:%.4f', sum(ccTest)./size(img_Test,4));
toc
%% Save Trained Net
[~,~]=mkdir('TrainedNet');
% save('TrainedNet/UNetT_mixedL_F64','net','info');
