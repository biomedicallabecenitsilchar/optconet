    function [accuracy] = error_rate(kernel_pars)
    % load P; load TV; load TVT; load T;

    load augmentedTrainingSet; load augmentedTestSet
    load YValidation;
    
     layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16,'Padding','same')
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
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

    options = trainingOptions('sgdm', ...
    'Momentum', kernel_pars(1),...
    'InitialLearnRate',kernel_pars(2), ...
    'MaxEpochs',ceil(kernel_pars(3)), ...
    'L2Regularization', kernel_pars(4),...    
    'Shuffle','every-epoch', ...
    'MiniBatchSize', 32,...
    'ValidationData',augmentedTestSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
     net = trainNetwork(augmentedTrainingSet,layers,options);

        [YPred] = classify(net,augmentedTestSet);
        

        accuracy = sum(YPred == YValidation)/numel(YValidation);



