% Author: Ma, yifan (yifan.ma@mailbox.tu-dresden.de)
% Date: 23.06.2021
% The MIT License
% Copyright (c) [2021] Ma yifan

clc
clear
close all
%% Load Dataset
load Dataset/mnist_32.mat
% img_Trn : Training images
% img_Vld : Validation images
% img_Test: Test images
% lbl_Trn : Training labels
% lbl_Vld : Validation labels
% lbl_Test: Test labels
inputsize=size(img_Trn,1:3);

% layers = 1;
dsXTrain = arrayDatastore(img_Trn,'IterationDimension',4);
dsYTrain = arrayDatastore(lbl_Trn,'IterationDimension',4);
dsTrain = combine(dsXTrain,dsYTrain);

dsXVld = arrayDatastore(img_Vld,'IterationDimension',4);
dsYVld = arrayDatastore(lbl_Vld,'IterationDimension',4);
dsVld = combine(dsXVld,dsYVld);

%% U-Net Layer
% lgraph = layerGraph();
% 
% tempLayers = [
%     imageInputLayer(inputsize,'Normalization','none',"Name","ImageInputLayer")
%     convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-1-ReLU-1")
%     convolution2dLayer([3 3],64,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-1-ReLU-2")
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2])
%     convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-2-ReLU-1")
%     convolution2dLayer([3 3],128,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-2-ReLU-2")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2])
%     convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-3-ReLU-1")
%     convolution2dLayer([3 3],256,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-3-ReLU-2")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2])
%     convolution2dLayer([3 3],512,"Name","Encoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-4-ReLU-1")
%     convolution2dLayer([3 3],512,"Name","Encoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Encoder-Stage-4-ReLU-2")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool","Stride",[2 2])
%     convolution2dLayer([3 3],1024,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Bridge-ReLU-1")
%     convolution2dLayer([3 3],1024,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Bridge-ReLU-2")
%     transposedConv2dLayer([2 2],512,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-UpReLU")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation")
%     convolution2dLayer([3 3],512,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-ReLU-1")
%     convolution2dLayer([3 3],512,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-1-ReLU-2")
%     transposedConv2dLayer([2 2],256,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-UpReLU")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
%     convolution2dLayer([3 3],256,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-ReLU-1")
%     convolution2dLayer([3 3],256,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-2-ReLU-2")
%     transposedConv2dLayer([2 2],128,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2],"WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-3-UpReLU")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%     depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
%     convolution2dLayer([3 3],128,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-3-ReLU-1")
%     convolution2dLayer([3 3],128,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-3-ReLU-2")
%     convolution2dLayer([3 3],64,"Name","Decoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-4-ReLU-1")
%     convolution2dLayer([3 3],64,"Name","Decoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
%     reluLayer("Name","Decoder-Stage-4-ReLU-2")
%     convolution2dLayer([1 1],1,"Name","Final-ConvolutionLayer")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% % clear up helper variable
% clear tempLayers;
% 
% lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
% lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
% lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
% lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
% lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Encoder-Stage-4-MaxPool");
% lgraph = connectLayers(lgraph,"Encoder-Stage-4-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
% lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","Decoder-Stage-1-DepthConcatenation/in1");
% lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","Decoder-Stage-2-DepthConcatenation/in1");
% lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","Decoder-Stage-3-DepthConcatenation/in1");

load Network/UNetCus_32.mat
dlnet=dlnetwork(lgraph);

%%
numEpochs = 25;
miniBatchSize = 40;
initialLearnRate = 1e-4;
%%
mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',@preprocessMiniBatch,...     
    'MiniBatchFormat',{'SSCB',''});

mbqV = minibatchqueue(dsVld,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',@preprocessMiniBatch,...     
    'MiniBatchFormat',{'SSCB',''});
%%
figure
lineLossTrain = animatedline('Color','#D95319','LineWidth',1);
lineLossValid = animatedline('Color','#0072BD','LineWidth',1);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
%% Training Loop
iteration = 0;
start = tic;
averageGrad = [];
averageSqGrad = [];

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX, dlY] = next(mbq);
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
%       learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the ADAM optimizer.
        [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
                                                       initialLearnRate);
        
        Validloss = modelPredictions(dlnet,mbqV);
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,loss)
        addpoints(lineLossValid,iteration,Validloss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

%% Evaluate
Label=lbl_Test; 
Pre=zeros(size(lbl_Test));
ccTest = [];
ssimval = [];
peaksnr = [];
for i =1:size(img_Test,4)
    dlYPred= predict(dlnet,dlarray(img_Test(:,:,1,i),'SSCB'));
    Pre(:,:,:,i) = dlYPred;
    Pre(:,:,:,i)=double(gather(extractdata(dlYPred)));
    %NPCC
    rou=corrcoef(Label(:,:,1,i),Pre(:,:,1,i));
    ccTest=[ccTest rou(1,2)];
    %SSIM
    ssimvaltemp=ssim(Label(:,:,1,i),Pre(:,:,1,i));
    ssimval = [ssimval ssimvaltemp];
    disp(i)
    %PSNR
    psnrtemp= psnr(Label(:,:,1,i),Pre(:,:,1,i));
    peaksnr =[peaksnr psnrtemp];
end

fprintf('NPCC of Test dataset is:%.3f', sum(ccTest)./size(img_Test,4));
fprintf('\n');
fprintf('SSIM of Test dataset is:%.3f', sum(ssimval)./size(img_Test,4));
fprintf('\n');
fprintf('PSNR of Test dataset is:%.3f', sum(peaksnr)./size(img_Test,4));
fprintf('\n');
%% Save Trained Net
[~,~]=mkdir('TrainedNet');
% save('TrainedNet/UNetT_mnist_16','dlnet');

%% Test Example
figure
tiledlayout(2,5,'TileSpacing','none')
for i=1:5
    nexttile
    imshow(Label(:,:,1,i),[0 255])
end
title('Label')
for j=1:5
    nexttile
    imshow(Pre(:,:,1,j),[0 255])
end
title('Prediction')

%% Loss Function

% function loss = LossFcn(Yp,Y)
%     loss = mse(Yp,Y)./size(Y,4);
% end

% normolized mse
function loss = LossFcn(Yp,Y)

for i = 1:size(Y,4)
    Y(:,:,1,i)=Y(:,:,1,i)./sqrt(sum(Y(:,:,1,i).^2,'all'));
    Yp(:,:,1,i)=Yp(:,:,1,i)./sqrt(sum(Yp(:,:,1,i).^2,'all'));
end

loss = mse(Yp,Y)./size(Y,4);
end
%% Calculate Gradient 
function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = LossFcn(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end
%% Validation Loss
function loss = modelPredictions(dlnet,mbq)
    shuffle(mbq)
    
    [dlXValid,dlYValid] = next(mbq);
    
    dlYPred = predict(dlnet,dlXValid);
    loss = LossFcn(dlYPred,dlYValid);
    loss = double(gather(extractdata(loss)));
end
%% minibatch process
function [X,Y] = preprocessMiniBatch(XCell,YCell)
    
    % Extract image data from cell and concatenate
    X = cat(4,XCell{:});
    % Extract label data from cell and concatenate
    Y = cat(4,YCell{:});

end