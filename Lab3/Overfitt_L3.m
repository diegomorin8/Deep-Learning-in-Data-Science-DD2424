%% Check gradients - Overfitting
%
% The first step is to prepare the workspace

%Prepare the workspace
close all;
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

%% Weights initialization

lambda = 0; 

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

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%% 2.7.3 Rho = 0.9 - No decay
%
% The parameters that we can set are chosen accordingly to the lecture
% notes
GDparams.n_batch = 100;
GDparams.eta = 0.001;
GDparams.n_epochs = 900;
rho = 0.95;
eta_decay = 1;
lambda = 0;

%%
%
% The main algorithm is applied. We need to output the costs and loss to
% plot them later. We are using just a small subset of the dataset
tic
[Wstar,bstar, val_loss_rho95, train_loss_rho95] = MiniBatchGDNorm(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
time_rho95 = toc;
%% Check performance of gradient descent - Accuracy test
%
% Using the test data, we want to check the performance of our netwoek
acc_rho95 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

% That of course, is really low. 


%% 2.7.7 Results Comparison

% We want to visualize the results of the trining. First we are plotting
% the training and validation loss for each epoch. 

figure; 
plot(val_loss_rho95);
hold on
plot(train_loss_rho95);
axis([0 200 0 4])
legend('Validation data', 'Training data')
title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.95, 1, time_rho95))
