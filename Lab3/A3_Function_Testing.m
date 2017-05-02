%% Assignment 2 
%
%% Introduction
%
% In this assignment a two layer network with multiple outputs will be
% trained to classify images from the dataset CIFAR-10. The network will be
% trained using minibatch gradient descent applied to a cost function that
% computes the cross-entropy loss of the classifier applied to the labelled
% training data and an L2 regularization term on the weight matrix.
%
% All the developed code will be implemented in this script, being
% separatedly in subsection that can be runned separately to see better the
% perfomance of the code.  

%% 1. Implementation
%
% The first step is to prepare the workspace

%Prepare the workspace
close all;
clear all;
clc;

%% Load the data
% 
% We will use the images in data_batch_1.mat as the training data. However
% this data has to be modified and adapted to our requirements. This is
% done by the function LoadBatch. This function loads the data we need, and
% outputs 3 parameters: 
%   - X: contains the image pixel data, has size dxN, is of type double or
%     single and has entries between 0 and 1. N is the number of images
%     (10000) and d the dimensionality of each image (3072=32×32×3).
%   - Y: is K×N (K= # of labels = 10) and contains the one-hot representation
%     of the label for each image.
%   - N: is a vector of length N containing the label for each image. A note
%     of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%     Matlab indexes matrices and vectors starting at 1. Therefore it may be
%     easier to encode the labels between 1-10.

[X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%% Pre - processing

mean_ = mean(X_batch, 2);
X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);


%% Gradients Check

lambda = 0; 

num_checks = 100; 
X_train = X_batch;
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30]; 

% Number of labels
[NumLabels, ~] = size(Y_train);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);



%% Check function init
% Weight and bias initialization
[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);
[W2, b2, mW2, mb2] = network_init2(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);
disp(norm(W2{1}-W{1}));
disp(norm(b2{1}-b{1}));
disp(norm(mW2{1}-mW{1}));
disp(norm(mb2{1}-mb{1}));
disp(norm(W2{2}-W{2}));
disp(norm(b2{2}-b{2}));
disp(norm(mW2{2}-mW{2}));
disp(norm(mb2{2}-mb{2}));


%% Check function - Evaluate Classifier
[P,h,s] = EvaluateClassifier(X_train, W, b);
[P2,h2,s2] = EvaluateClassifier2(X_train, W, b);
disp(sum(sum(P - P2)));
disp(norm(h2 - h{2}));
disp(norm(s2 - s{1}));

%% Check function - Evaluate Classifier
J = ComputeCost(X_train, Y_train, W, b, lambda);
J2 = ComputeCost2(X_train, Y_train, W, b, lambda);
disp(J - J2);

%% Check function - ComputeGradients
[grad_W, grad_b] = ComputeGradients(X_train, Y_train, P, W, h, s, lambda);
[grad_W2, grad_b2] = ComputeGradients2(X_train, Y_train, P, W, h2, s2, lambda);
disp(norm(grad_W2{1}-grad_W{1}));
disp(norm(grad_b2{1}-grad_b{1}));
disp(norm(grad_W2{2}-grad_W{2}));
disp(norm(grad_b2{2}-grad_b{2}));

%% Check function - ComputeGradients
acc = ComputeAccuracy(X_test, y_test, W, b);
acc2 = ComputeAccuracy2(X_test, y_test, W, b);
disp(norm(acc - acc2));
%% Check Gradients

%Prepare the workspace
close all;
clear all;
clc;

[X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%% Pre - processing

mean_ = mean(X_batch, 2);
X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

%% Gradients Check - 2 layers

lambda = 0; 

num_images = 5; 
size_image = 100; 
X_train = X_batch(1:size_image,1:num_images);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50]; 

% Number of labels
[NumLabels, ~] = size(Y_train);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_train(:,i), Y_train(:,i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s] = EvaluateClassifier(X_train(:,i), W, b);
    % Analytical batch
    [grad_W, grad_b] = ComputeGradients(X_train(:,i), Y_train(:,i), P, W, h, s, lambda);
    
    % Calculate the error according to the lab notes
    error_W(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    % The error has to be small enough
    if error_W(i) >= 1e-8
        disp(' Error in W gradient calculation ')
        
    elseif error_b(i) >= 1e-8
        disp(' Error in b gradient calculation ')
        
    end
end

%%
figure;
plot(error_W)
hold on
plot(error_b)
legend( 'Error in W', ' Error in B')
axis([ 1 5 0 1e-10 ])

%% Gradients Check - 3 layers

lambda = 0; 

num_images = 5; 
size_image = 100; 
X_train = X_batch(1:size_image,1:num_images);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30]; 

% Number of labels
[NumLabels, ~] = size(Y_train);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_train(:,i), Y_train(:,i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s] = EvaluateClassifier(X_train(:,i), W, b);
    % Analytical batch
    [grad_W, grad_b] = ComputeGradients(X_train(:,i), Y_train(:,i), P, W, h, s, lambda);
    
    % Calculate the error according to the lab notes
    error_W(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    % The error has to be small enough
    if error_W(i) >= 1e-8
        disp(' Error in W gradient calculation ')
        
    elseif error_b(i) >= 1e-8
        disp(' Error in b gradient calculation ')
        
    end
end

%%
figure;
plot(error_W)
hold on
plot(error_b)
legend( 'Error in W', ' Error in B')
axis([ 1 5 0 1e-9 ])

%% Gradients Check - 4 layers

lambda = 0; 

num_images = 5; 
size_image = 100; 
X_train = X_batch(1:size_image,1:num_images);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30 20]; 

% Number of labels
[NumLabels, ~] = size(Y_train);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_train(:,i), Y_train(:,i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s] = EvaluateClassifier(X_train(:,i), W, b);
    % Analytical batch
    [grad_W, grad_b] = ComputeGradients(X_train(:,i), Y_train(:,i), P, W, h, s, lambda);
    
    % Calculate the error according to the lab notes
    error_W(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    % The error has to be small enough
    if error_W(i) >= 1e-8
        disp(' Error in W gradient calculation ')
        
    elseif error_b(i) >= 1e-8
        disp(' Error in b gradient calculation ')
        
    end
end

%%
figure;
plot(error_W)
hold on
plot(error_b)
legend( 'Error in W', ' Error in B')
axis([ 1 5 0 1e-9])


%% Check gradients - Overfitting
%
%Prepare the workspace
close all;
clear all;
clc;

[X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%% Pre - processing

mean_ = mean(X_batch, 2);
X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

%% Gradients Check Overfitting - 2 layers

lambda = 0; 

num_images = 5; 
size_image = 100; 
X_train = X_batch(1:size_image,1:num_images);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50]; 

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
%
% The parameters that we can set are chosen accordingly to the lecture
% notes
GDparams.n_batch = 100;
GDparams.eta = 0.35;
GDparams.n_epochs = 200;
lambda = 0;

%%
%
% The main algorithm is applied. We need to output the costs and loss to
% plot them later. We are using just a small subset of the dataset
tic
[Wstar, bstar, val_loss_no2, train_loss_no2] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2);
time_noM = toc;
%% Check performance of gradient descent - Accuracy test
%
% Using the test data, we want to check the performance of our netwoek
acc_no2 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

% That of course, is really low.

%% Gradients Check Overfitting - 2 layers

lambda = 0; 

num_images = 5; 
size_image = 100; 
X_train = X_batch(1:size_image,1:num_images);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30]; 

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
%
% The parameters that we can set are chosen accordingly to the lecture
% notes
GDparams.n_batch = 100;
GDparams.eta = 0.035;
GDparams.n_epochs = 200;
lambda = 0;

%%
%
% The main algorithm is applied. We need to output the costs and loss to
% plot them later. We are using just a small subset of the dataset
tic
[Wstar, bstar, val_loss_no3, train_loss_no3] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2);
time_noM = toc;
%% Check performance of gradient descent - Accuracy test
%
% Using the test data, we want to check the performance of our netwoek
acc_no3 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

% That of course, is really low.

%% Gradients Check Overfitting - 2 layers

lambda = 0; 

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30 20]; 

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
%
% The parameters that we can set are chosen accordingly to the lecture
% notes
GDparams.n_batch = 100;
GDparams.eta = 0.005;
GDparams.n_epochs = 200;
lambda = 0;

%%
%
% The main algorithm is applied. We need to output the costs and loss to
% plot them later. We are using just a small subset of the dataset
tic
[Wstar, bstar, val_loss_no4, train_loss_no4] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2);
time_noM = toc;
%% Check performance of gradient descent - Accuracy test
%
% Using the test data, we want to check the performance of our netwoek
acc_no4 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

% That of course, is really low.

%% 2.7.7 Results Comparison

% We want to visualize the results of the trining. First we are plotting
% the training and validation loss for each epoch. 

figure; 
subplot(3,1,1)
plot(val_loss_no2);
hold on
plot(train_loss_no2);
legend('Validation data', 'Training data')
title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - Num layers = %g', GDparams.eta, 0, 1, 2))
hold off
subplot(3,1,2)
plot(val_loss_no3);
hold on
plot(train_loss_no3);
legend('Validation data', 'Training data')
title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - Num layers = %g', GDparams.eta, 0.5, 1, 3))
hold off
subplot(3,1,3)
plot(val_loss_no4);
hold on
plot(train_loss_no4);
legend('Validation data', 'Training data')
title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - Num layers = %g', GDparams.eta, 0.95, 1, 4))
hold off

%% 3.1 Reasonable range of values for the learning rate
%
% Steps: 
%   1. Set the regularization term to a small value (say .000001.)

%   2. Randomly initalized the network. It should perform similarly to a random
%      guesser and thus the training loss before you start learning should 
%      be ?2.3 (? ln(.1)). 

% We want to clean the work space to start the experiments again

clc;
close all; 

lambda = 0.000001;


%%% Load the data
% 
% We will use the images in data_batch_1.mat as the training data. However
% this data has to be modified and adapted to our requirements. This is
% done by the function LoadBatch. This function loads the data we need, and
% outputs 3 parameters: 
%   - X: contains the image pixel data, has size dxN, is of type double or
%     single and has entries between 0 and 1. N is the number of images
%     (10000) and d the dimensionality of each image (3072=32×32×3).
%   - Y: is K×N (K= # of labels = 10) and contains the one-hot representation
%     of the label for each image.
%   - N: is a vector of length N containing the label for each image. A note
%     of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%     Matlab indexes matrices and vectors starting at 1. Therefore it may be
%     easier to encode the labels between 1-10.

[X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%% Pre - processing

mean_ = mean(X_batch, 2);
X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

%% Weights initialization

error_W = [];
error_b = [];

num_checks = 100; 

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = 50;

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

% Weight and bias initialization
[W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

%%
%
% The parameters that we can set are chosen accordingly to the lecture
% notes

% Cost function
J = ComputeCost(X_batch, Y_batch, W, b, lambda);

%% 
%
%   3. You want to perform a quick search by hand to find the rough bounds 
%      for reasonable values of the learning rate. During this stage you 
%      should print out the training cost/lost after every epoch. If the 
%      learning rate is too small, then after each epoch you will find 
%      that the training loss barely changes. While if the learning rate is 
%      too large, then learning is unstable and you will probably get 
%      NaNs and/or very high loss values. You should only need ?5 epochs to
%      check these properties.


% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 2;
rho = 0.9;
eta_decay = 0.95;
valLoss = [];
trainLoss = [];
index = [];
cont = 0;
str = [];
for i = 0.01:0.02:0.16
    cont = cont + 1
    GDparams.eta = i;
    [~, ~, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    valLoss = [valLoss , val_loss];
    trainLoss = [trainLoss , train_loss];
    index = [index ; i];
    str = [str ; sprintf('eta = %g',i)];
end

%% Get the range of etas

figure; 
plot(trainLoss);
legend(str)
hold off

%%
rng(400)
%% Weights initialization

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = 50;

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

% Weight and bias initialization
[W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

%Valid ranges
minEta = 0.005;
maxEta = 0.3;

minL = 0.0005;
maxL = 0.01;

%% 
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 1;
rho = 0.9;
eta_decay = 0.95;
valLoss1 = [];
trainLoss1 = [];
etas1 = [];
lambdas1 = [];
accuracy1 = [];
cont  = 0;
for i = 1:60
    cont = cont + 1;
    e = minEta + (maxEta - minEta)*rand(1, 1); 
    etas1 = [etas1 ; e];
    GDparams.eta = etas1(i);
    l = minL + (maxL - minL)*rand(1, 1);
    lambdas1 = [lambdas1 ; l];
    lambda = lambdas1(i);
    [W_end, b_end, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
    valLoss1 = [valLoss1 , val_loss];
    trainLoss1 = [trainLoss1 , train_loss];
    acc = ComputeAccuracy(X_val, y_val, W_end, b_end);
    accuracy1 = [accuracy1, acc] ; 
    disp(sprintf('Cont = %i -  Accuracy = %g - Eta = %g - Lambda = %g', cont, accuracy1(i), etas1(i), lambdas1(i)))
end

%%
% valLoss1( valLoss1 == 0 ) = Inf; 
   
[ accVal1, accInd1] = sort( accuracy1, 'descend' ); 
[lossValues1, lossIndices1] = sort(valLoss1(end,1:end), 'ascend');

N = 15; 
accInd1 = accInd1(1:N);
lossIndices1 = lossIndices1(1:N);
eta_lambda_pair1 = [ etas1(accInd1) , lambdas1(accInd1)];


maxL1 = max(lambdas1(accInd1));
minL1 = min(lambdas1(accInd1));
maxEta1 = max(etas1(accInd1));
minEta1 = min(etas1(accInd1));

%%

rng(400)

num_checks = 100; 

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = 50;

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

% Weight and bias initialization
[W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 3;
rho = 0.9;
eta_decay = 0.95;
valLoss2 = [];
trainLoss2 = [];
etas2 = [];
lambdas2 = [];
accuracy2 = [];
cont = 0; 
minEta1 = minEta1 * 0.1;
minL1 = minL1 * 0.1;
for i = 1:60
    cont = cont + 1 ;
    e = minEta1 + (maxEta1 - minEta1)*rand(1, 1);
    etas2 = [etas2 ; e];
    GDparams.eta = etas2(i);
    l = minL1 + (maxL1 - minL1)*rand(1, 1);
    lambdas2 = [lambdas2 ; l];
    lambda = lambdas2(i);
    [W_end, b_end,  val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
    valLoss2 = [valLoss2 , val_loss];
    trainLoss2 = [trainLoss2 , train_loss];
    accuracy2 = [accuracy2, ComputeAccuracy(X_val, y_val, W_end, b_end)] ; 
    disp(sprintf('Cont = %i -  Accuracy = %g - Eta = %g - Lambda = %g', cont, accuracy2(i), etas2(i), lambdas2(i)))
end

%%
valLoss2( valLoss2 == 0 ) = Inf; 
   
[ accVal2, accInd2] = sort( accuracy2, 'descend' ); 
[lossValues2, lossIndices2] = sort(valLoss2(end,1:end), 'ascend');

N = 8; 
accInd2 = accInd2(1:N);
lossIndices2 = lossIndices2(1:N);
eta_lambda_pair2 = [ etas2(accInd2) , lambdas2(accInd2)];

maxL2 = max(lambdas2(accInd2));
minL2 = min(lambdas2(accInd2));
maxEta2 = max(etas2(accInd2));
minEta2 = min(etas2(accInd2));

%%
rng(400)
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
valLoss3 = [];
trainLoss3 = [];
etas3 = [];
lambdas3 = [];
accuracy3 = [];
cont = 0;
for i = 1:80
    cont = cont + 1
    e = minEta2 + (maxEta2 - minEta2)*rand(1, 1);
    etas3 = [etas3 ; e];
    GDparams.eta = etas3(i);
    l = minL2 + (maxL2 - minL2)*rand(1, 1);
    lambdas3 = [lambdas3 ; l];
    lambda = lambdas3(i);
    [W_end, b_end, val_loss, train_loss] = MiniBatchGD(X_batch(:,1:300), Y_batch(:,1:300), X_val(:,1:300), Y_val(:,1:300), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    valLoss3 = [valLoss3 , val_loss];
    trainLoss3 = [trainLoss3 , train_loss];
    etas3 = [etas3 ; e];
    lambdas3 = [lambdas3 ; l];
    accuracy3 = [accuracy3, ComputeAccuracy(X_test, y_test, W_end, b_end)] ; 
    disp(sprintf(' Accuracy = %g ', accuracy3(i)))
end

%%
valLoss3( valLoss3 == 0 ) = Inf; 
   
[ accVal3, accInd3] = sort( accuracy3, 'descend' ); 
[lossValues3, lossIndices3] = sort(valLoss3(end,1:end), 'ascend');

N = 10; 
accInd3 = accInd3(1:N);
lossIndices3 = lossIndices3(1:N);
eta_lambda_pair3 = [ etas3(accInd3) , lambdas3(accInd3)];

maxL3 = max(lambdas3(accInd3));
minL3 = min(lambdas3(accInd3));
maxEta3 = max(etas3(accInd3));
minEta3 = min(etas3(accInd3));

%%

eta_try = eta_lambda_pair1(1,1);
l_try = eta_lambda_pair1(1,2);

%%
num_checks = 100; 

% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = 50;

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

% Weight and bias initialization
[W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

GDparams.eta = 0.0404;
lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;
accuracyFinal = []; 
vLossFinal = [];
tLossFinal = [];
cont = 0; 
for i = 1:length(eta_lambda_pair2)
    cont = cont + 1; 
    GDparams.eta = eta_lambda_pair2(i,1);
    lambda = eta_lambda_pair2(i,2);
    [W_end, b_end, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
    accuracyFinal = [accuracyFinal, ComputeAccuracy(  X_test, y_test, W_end, b_end)];
    vLossFinal = [vLossFinal, val_loss];
    tLossFinal = [tLossFinal, train_loss];
    disp(sprintf('Cont = %i -  Accuracy = %g - Eta = %g - Lambda = %g', cont, accuracyFinal(i), GDparams.eta, lambda))
end

%%
   
[ accValFinal, accIndFinal] = sort( accuracyFinal, 'descend' ); 

N = 8; 
accIndFinal = accIndFinal(1:N);
eta_lambda_pairFinal = [ eta_lambda_pair2(accIndFinal,1) , eta_lambda_pair2(accIndFinal,2)];
eta_pair_accu = [ eta_lambda_pair2(accIndFinal,1) , eta_lambda_pair2(accIndFinal,2), accuracyFinal(accIndFinal)']
%%

figure;
plot(val_loss)
hold on
plot(train_loss)

%%

%%

% Load data sets
[ X_train, Y_train, y_train ] = LoadBatch( 'data_batch_1.mat' );
[ X_val, Y_val, y_val ] = LoadBatch( 'data_batch_2.mat' );
[ X_train2, Y_train2, y_train2 ] = LoadBatch( 'data_batch_3.mat' );
[ X_train3, Y_train3, y_train3 ] = LoadBatch( 'data_batch_4.mat' );
[ X_train4, Y_train4, y_train4 ] = LoadBatch( 'data_batch_5.mat' );
[ X_test, Y_test, y_test ] = LoadBatch( 'test_batch.mat' );

% We take 9000 of the samples from the validation set
X_train = [X_train, X_val(:, 1:9000), X_train2, X_train3, X_train4];
Y_train = [Y_train, Y_val(:, 1:9000), Y_train2, Y_train3, Y_train4];
y_train = [y_train; y_val(1:9000); y_train2; y_train3; y_train4];
clear X_train2 X_train3 X_train4
clear Y_train2 Y_train3 Y_train4
clear y_train2 y_train3 y_train4

X_val = X_val(:, 9001:end);
Y_val = Y_val(:, 9001:end);
y_val = y_val(9001:end);

mean_ = mean(X_train, 2);
X_train = X_train - repmat(mean_, [1, size(X_train, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);
% Number of features 

[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = 50;

% Number of labels
[NumLabels, ~] = size(Y_train);

% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

% Weight and bias initialization
[W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 30;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = eta_lambda_pairFinal(1,1);
lambda = eta_lambda_pairFinal(1,2);
[W_end, b_end, val_loss, train_loss] = MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
save('Big_train.mat','val_loss','train_loss','W_end','b_end')
%%
save('Fine_tuning_pairs.mat','eta_lambda_pair1','eta_lambda_pair2','accuracyFinal','vLossFinal', 'tLossFinal')

%%
for i = 1:3
    figure;
    plot(vLossFinal(:,accIndFinal(i)))
    hold on
    plot(tLossFinal(:,accIndFinal(i)))
    title(sprintf( 'Validation vs training loss for eta = %g - lambda = %g. Final accuracy = %g', eta_lambda_pairFinal(i,1), eta_lambda_pairFinal(i,2), accValFinal(i))) 
    legend('Validation loss', 'Training loss' )
    hold off
end

%%
=

    0.0858    0.0004
    0.0501    0.0024
    0.0319    0.0043
    0.0613    0.0015
    0.0683    0.0010
    0.0386    0.0057
    0.0447    0.0037
    
    eta_lambda_pair1 =

    0.0957    0.0045
    0.1068    0.0031
    0.0439    0.0013
    0.1024    0.0087
    0.0695    0.0052
    0.1198    0.0077
    0.0439    0.0033
    0.1442    0.0016
    0.0741    0.0068
    0.0592    0.0056
    0.0608    0.0061
    0.0515    0.0077
    0.1298    0.0021
    0.0537    0.0078
    0.0558    0.0060
    0.0390    0.0061