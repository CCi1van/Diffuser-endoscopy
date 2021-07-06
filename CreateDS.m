% Author: Ma, yifan (yifan.ma@mailbox.tu-dresden.de)
% Date: 17.06.2021
% The MIT License
% Copyright (c) [2021] Ma yifan
clc
clear
close all

%% Variables Setting
layers=1; % depth layers
p_Trn =0.8; % proportion of traning in a_Trn

%% Load raw Training Dataset
load Dataset/mnist_L_trn.mat

a_Trn=size(Images,4); % amount for training and validation


img_Trn   = Images(:,:,layers,1:p_Trn*a_Trn);
img_Vld  = Images(:,:,layers,p_Trn*a_Trn+1:a_Trn);
lbl_Trn  = Labels(:,:,layers,1:p_Trn*a_Trn);
lbl_Vld  = Labels(:,:,layers,p_Trn*a_Trn+1:a_Trn);


%% Load raw Test Dataset
load Dataset/mnist_L_test.mat

a_Test=size(Images,4); % amount for test
img_Test  = Images;
lbl_Test  = Labels;

%% Create mixed Dataset

% idx=randperm(a_Trn);
% Tidx=randperm(a_Test);

% load Dataset/mnist_L_trn.mat
% img_temp = Images;
% lbl_temp = Labels;
% load Dataset/fashion_L_trn.mat
% img = cat(4,Images,img_temp);
% lbl = cat(4,Labels,lbl_temp);
% 
% img = img(:,:,layers,idx);
% lbl = lbl(:,:,layers,idx);
% 
% img_Trn   = img(:,:,layers,1:p_Trn*a_Trn);
% img_Vld  = img(:,:,layers,p_Trn*a_Trn+1:a_Trn);
% lbl_Trn  = lbl(:,:,layers,1:p_Trn*a_Trn);
% lbl_Vld  = lbl(:,:,layers,p_Trn*a_Trn+1:a_Trn);
% 
% load Dataset/mnist_L_test.mat
% img_temp = Images;
% lbl_temp = Labels;
% load Dataset/fashion_L_test.mat
% img = cat(4,Images,img_temp);
% lbl = cat(4,Labels,lbl_temp);
% 
% img = img(:,:,layers,Tidx);
% lbl = lbl(:,:,layers,Tidx);
% 
% img_Test  = img;
% lbl_Test  = lbl;

%% Figure Example
figure
tiledlayout(2,5,'TileSpacing','none')
for i=1:5
    nexttile
    imshow(img_Trn(:,:,1,i),[0 255])
end
title('Input')
for j=1:5
    nexttile
    imshow(lbl_Trn(:,:,1,j),[0 255])
end
title('Label')

%% Normalization
for i = 1:size(img_Trn,4)
    img_Trn(:,:,1,i)=img_Trn(:,:,1,i)./sqrt(sum(img_Trn(:,:,1,i).^2,'all'));
    lbl_Trn(:,:,1,i)=lbl_Trn(:,:,1,i)./sqrt(sum(lbl_Trn(:,:,1,i).^2,'all'));
end
for i = 1:size(img_Vld,4)
    img_Vld(:,:,1,i)=img_Vld(:,:,1,i)./sqrt(sum(img_Vld(:,:,1,i).^2,'all'));
    lbl_Vld(:,:,1,i)=lbl_Vld(:,:,1,i)./sqrt(sum(lbl_Vld(:,:,1,i).^2,'all'));
end
for i = 1:size(img_Test,4)
    img_Test(:,:,1,i)=img_Test(:,:,1,i)./sqrt(sum(img_Test(:,:,1,i).^2,'all'));
    lbl_Test(:,:,1,i)=lbl_Test(:,:,1,i)./sqrt(sum(lbl_Test(:,:,1,i).^2,'all'));
end
%% Save Dataset
[~,~]=mkdir('Dataset');
save('Dataset/mnistL_Norm','img_Trn','img_Vld','img_Test','lbl_Trn','lbl_Vld','lbl_Test');
