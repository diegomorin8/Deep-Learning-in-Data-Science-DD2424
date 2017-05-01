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

rng(400)
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

%% Gradients Check Overfitting - 2 layers


% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50]; 

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%% STEP 1: REPLICATE RESULTS
%
% The parameters that we can set are chosen accordingly to the lecture
% notes
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
GDparams.eta = 0.0319;
lambda = 0.0043;
eta_decay = 0.95;
rho = 0.9;
%% Check if with 2 layers we get the same result
%
% The main algorithm is applied. We need to output the costs and loss to
% plot them later. We are using just a small subset of the dataset
[W_end, b_end, val_loss, train_loss] = MiniBatchGD2(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%% Check performance of gradient descent - Accuracy test
%
% Using the test data, we want to check the performance of our netwoek
acc = ComputeAccuracy(X_test, y_test, W_end, b_end);


%% Plotting 

figure
plot(val_loss);
hold on
plot(train_loss);
legend('Validation' ,'Training', acc2)
title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - Num layers = %g', GDparams.eta, 0.5, 1, 3))
