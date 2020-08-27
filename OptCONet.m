clc; 
% Add the path of the Files

digitDatasetPath = fullfile('C:\Users\MIPLAB\Documents\Covid\XRay\COVIDDatabase');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2});

start_time_train=cputime;
% net = resnet50();

no_person = 3;

%splitting of Database into Training and Test Sets

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

% Data Augmentation
    
imageSize = [224 224 3];
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

% Design CNN Network

layers = [
imageInputLayer([224 224 3])

convolution2dLayer(7,128,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(5,128,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(5,64,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,64,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer

maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer

fullyConnectedLayer(3)
softmaxLayer
classificationLayer];

YValidation = testSet.Labels;


save augmentedTrainingSet augmentedTrainingSet;
save  augmentedTestSet  augmentedTestSet;
save YValidation YValidation

% Initialize the parameters for WOA Search

SearchAgents_no=30;   Max_iteration=5; dim = 4;
lb =  [0.5, 0.01, 5, 1.0000e-04];
ub = [0.9, 0.1, 10, 5.0000e-04]; 

% WOA Search Optimization

[Best_score,Best_pos,GWO_cg_curve] = GWO(SearchAgents_no,Max_iteration,lb,ub,dim,@error_rate);
K1 = Best_pos(1,1);
K2 = Best_pos(1,2);
K3 = Best_pos(1,3);
K4 = Best_pos(1,4);


        options = trainingOptions('sgdm', ...
    'Momentum', K1,...
    'InitialLearnRate',K2, ...
    'MaxEpochs',ceil(K3), ...
    'L2Regularization', K4,...    
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedTestSet, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the CNN network, Parameters optimized using WOA 

net = trainNetwork(augmentedTrainingSet,layers,options);

[YPred, Scores] = classify(net,augmentedTestSet);

accuracy = sum(YPred == YValidation)/numel(YValidation);

TestError = (1-accuracy)*100;
    
   
     
    
    
     
     
   

