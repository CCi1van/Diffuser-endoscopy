% Script to create training data for 3D image reconstruction from speckle
% images with a 2D convolutional neural network with depth position as
% intensity
% 
% Author: Tom Glosemeyer (tom.glosemeyer@tu-dresden.de)
% Date: 26.04.2021

% modified by Ma, yifan (yifan.ma@mailbox.tu-dresden.de)
% The MIT License (MIT) 
% Copyright (c) [2021] Ma yifan

% The input size of object plane is 64x64
% the output size of camera is 128x128

clc
clear
close all



%% Variables
layers=1; % depth layers
I_pix=128; % pixel Images
L_pix=64; % pixel Labels
opl=1000; % number of training objects with x=1...layers occupied layers
o_max=10000; % number of different object images (digits) -> do not change unless other Truth images are used
displace=1; % 1 for displaced digits
resize_scale=0.5; % resize scale max. 2
relu=1; % 1 for set (elements = thr) = 0
thr=30; % background value
%% File path setting
path=fullfile('DNN_Reshape_128','8_zdiff=400um'); % path to speckle images
obj_layers = 8; % the layers where objects lay in
% path2=fullfile('MNIST-Dataset','mnist','TrainingData'); % path to digit imagess
path2=fullfile('MNIST-Dataset','mnist','TestData'); % path to digit imagess
imgList=dir(fullfile(path2,'**/*.png'));
%% Create Training Objects in multiple layers
o_tot=opl*layers;  % total training objects
Images=zeros(I_pix,I_pix,layers,o_tot); % Network imput(speckle patterns)
Labels=zeros(L_pix,L_pix,layers,o_tot);% Network output(ground-truth images)
num_o = zeros(layers,o_max);
for II=1:layers
    num_o(II,:)=randperm(o_max); % random order of digits
end
for I=1:o_tot
    l_mix = randperm(layers); % random order of layers to occupy
    for II=1:floor((I-1)/opl)+1 % number of occupied layers
        num = num_o(II,mod(I,o_max)+1);
        digimg=imread(fullfile(imgList(num).folder,imgList(num).name)); % load digit image
        digres=rgb2gray(digimg);
        if relu ==1
            digres(digres==thr)=0;
        end
        % resize
            digres=imresize(digres,resize_scale);
            padsize =L_pix-size(digres);
            Labels(:,:,l_mix(II),I)=padarray(digres,padsize,0,'post');

        if displace==1 % random displacement of digit image for data augmentation
            Labels(:,:,l_mix(II),I)=circshift(Labels(:,:,l_mix(II),I),[randi([1 L_pix]) randi([1 L_pix])]);
        end
  
    end
    % create speckle pattern for digit image
    for ij=0:L_pix-1
        for ii=0:L_pix-1
            for III=1:layers
                if Labels(ii+1,ij+1,III,I)>0
                    num = num2str((III-1)*L_pix^2+ij*L_pix+ii+(obj_layers-1)*4096+1);
                    while length(num)<4
                        num = ['0' num]; % correct file name for point speckle pattern
                    end
                    Images(:,:,III,I) = Images(:,:,III,I) + Labels(ii+1,ij+1,III,I)*double(imread(fullfile(path,strcat(num,'.png'))));
                end
            end
        end
    end
    Images(:,:,III,I) = 255*Images(:,:,III,I)/max(max(Images(:,:,III,I))); % normalization of speckle pattern image
    
    disp(I);
    
end


%% Example
figure
subplot(1,2,1)
imshow(Images(:,:,1,1),[0 255])
title('Input')
subplot(1,2,2)
imshow(Labels(:,:,1,1),[0 255])
title('Label')

%% Save Dataset
[~,~]=mkdir('Dataset');
save('Dataset/mnist_0.5_test_.mat','Images','Labels');% Change the file name for different data
