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

lambda = 0; 

num_images = 51; 
size_image = 50; 
X_train = X_batch(1:size_image,:);
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

%% STEP 1: REPLICATE RESULTS
%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlowNorm(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s, s_norm, mu_, v] = ForwardBatch(X_train(:,1+num_images*(i-1):num_images*i), W, b);
    % Analytical batch
    [grad_W, grad_b] = BackwardBN(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), P, W, h, s, s_norm, mu_, v, lambda);
    
    % Calculate the error according to the lab notes
    error_W1(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b1(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    error_W2(i) = norm(grad_W{2} - ngrad_W{2})/(norm(grad_W{2}) + norm(ngrad_W{2}));
    error_b2(i) = norm(grad_b{2} - ngrad_b{2})/(norm(grad_b{2}) + norm(ngrad_b{2}));
    
    
    % The error has to be small enough
    if error_W1(i) >= 1e-8
        disp(' Error in W1 gradient calculation ')
        
    elseif error_b1(i) >= 1e-8
        disp(' Error in b1 gradient calculation ')
        
    end
    
    % The error has to be small enough
    if error_W2(i) >= 1e-8
        disp(' Error in W2 gradient calculation ')
        
    elseif error_b2(i) >= 1e-8
        disp(' Error in b2 gradient calculation ')
        
    end
end

%%
figure;
plot(error_W1)
hold on
plot(error_b1)
hold on
plot(error_W2)
hold on
plot(error_b2)
legend( 'Error in W - L1', ' Error in B - L1', 'Error in W - L2', ' Error in B - L2', 'Error in W - L3', ' Error in B - L3', 'Error in W - L4', ' Error in B - L4')


%% Gradients Check Overfitting - 2 layers

lambda = 0; 

num_images = 51; 
size_image = 50; 
X_train = X_batch(1:size_image,:);
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

%% STEP 1: REPLICATE RESULTS
%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlowNorm(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s, s_norm, mu_, v] = ForwardBatch(X_train(:,1+num_images*(i-1):num_images*i), W, b);
    % Analytical batch
    [grad_W, grad_b] = BackwardBN(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), P, W, h, s, s_norm, mu_, v, lambda);
    
    % Calculate the error according to the lab notes
    error_W1(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b1(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    error_W2(i) = norm(grad_W{2} - ngrad_W{2})/(norm(grad_W{2}) + norm(ngrad_W{2}));
    error_b2(i) = norm(grad_b{2} - ngrad_b{2})/(norm(grad_b{2}) + norm(ngrad_b{2}));
    
    error_W3(i) = norm(grad_W{3} - ngrad_W{3})/(norm(grad_W{3}) + norm(ngrad_W{3}));
    error_b3(i) = norm(grad_b{3} - ngrad_b{3})/(norm(grad_b{3}) + norm(ngrad_b{3}));
    
    % The error has to be small enough
    if error_W1(i) >= 1e-8
        disp(' Error in W gradient calculation ')
        
    elseif error_b1(i) >= 1e-8
        disp(' Error in b gradient calculation ')
        
    end
    
    % The error has to be small enough
    if error_W2(i) >= 1e-8
        disp(' Error in W 3 gradient calculation ')
        
    elseif error_b2(i) >= 1e-8
        disp(' Error in b 3 gradient calculation ')
        
    end
    
    % The error has to be small enough
    if error_W3(i) >= 1e-8
        disp(' Error in W 3 gradient calculation ')
        
    elseif error_b3(i) >= 1e-8
        disp(' Error in b 3 gradient calculation ')
        
    end
end

%%
figure;
plot(error_W1)
hold on
plot(error_b1)
hold on
plot(error_W2)
hold on
plot(error_b2)
hold on
plot(error_W3)
hold on
plot(error_b3)
legend( 'Error in W - L1', ' Error in B - L1', 'Error in W - L2', ' Error in B - L2', 'Error in W - L3', ' Error in B - L3')
axis([ 1 5 0 1e-9 ])

%% Gradients Check Overfitting - 2 layers

lambda = 0; 

num_images = 51; 
size_image = 50; 
X_train = X_batch(1:size_image,:);
Y_train = Y_batch;

% Number of features 
[DataDim, ~] = size(X_train);

% Number of nodes in the hidden layer
HIDDEN_NODES = [20 10 5]; 

% Number of labels
[NumLabels, ~] = size(Y_train);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_train,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%% STEP 1: REPLICATE RESULTS
%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlowNorm(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s, s_norm, mu_, v] = ForwardBatch(X_train(:,1+num_images*(i-1):num_images*i), W, b);
    % Analytical batch
    [grad_W, grad_b] = BackwardBN(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), P, W, h, s, s_norm, mu_, v, lambda);
    
    % Calculate the error according to the lab notes
    error_W1(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
    error_b1(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));
    
    error_W2(i) = norm(grad_W{2} - ngrad_W{2})/(norm(grad_W{2}) + norm(ngrad_W{2}));
    error_b2(i) = norm(grad_b{2} - ngrad_b{2})/(norm(grad_b{2}) + norm(ngrad_b{2}));
    
    error_W3(i) = norm(grad_W{3} - ngrad_W{3})/(norm(grad_W{3}) + norm(ngrad_W{3}));
    error_b3(i) = norm(grad_b{3} - ngrad_b{3})/(norm(grad_b{3}) + norm(ngrad_b{3}));
    
    error_W4(i) = norm(grad_W{4} - ngrad_W{4})/(norm(grad_W{4}) + norm(ngrad_W{4}));
    error_b4(i) = norm(grad_b{4} - ngrad_b{4})/(norm(grad_b{4}) + norm(ngrad_b{4}));
    
    % The error has to be small enough
    if error_W1(i) >= 1e-8
        disp(' Error in W gradient calculation ')
        
    elseif error_b1(i) >= 1e-8
        disp(' Error in b gradient calculation ')
        
    end
    
    % The error has to be small enough
    if error_W2(i) >= 1e-8
        disp(' Error in W 3 gradient calculation ')
        
    elseif error_b2(i) >= 1e-8
        disp(' Error in b 3 gradient calculation ')
        
    end
    
    % The error has to be small enough
    if error_W3(i) >= 1e-8
        disp(' Error in W 3 gradient calculation ')
        
    elseif error_b3(i) >= 1e-8
        disp(' Error in b 3 gradient calculation ')
        
    end
    
    if error_W4(i) >= 1e-8
        disp(' Error in W 3 gradient calculation ')
        
    elseif error_b4(i) >= 1e-8
        disp(' Error in b 3 gradient calculation ')
        
    end
end

%%
figure;
plot(error_W1)
hold on
plot(error_b1)
hold on
plot(error_W2)
hold on
plot(error_b2)
hold on
plot(error_W3)
hold on
plot(error_b3)
hold on
plot(error_W4)
hold on
plot(error_b4)
legend( 'Error in W - L1', ' Error in B - L1', 'Error in W - L2', ' Error in B - L2', 'Error in W - L3', ' Error in B - L3', 'Error in W - L4', ' Error in B - L4')
axis([ 1 5 0 1e-9 ])

