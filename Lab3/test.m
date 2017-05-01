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

num_images = 10; 
size_image = 100; 
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

%%
% As we are going to use different mini batches to check the performance,
% we want a loop. 
for i = 1:num_images
    
    disp(sprintf(' Iteration = %i', i))
    % Numerical batch
    [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), W, b, lambda, 1e-5);
    
    % We need P
    [P, h, s, s_norm, mu_, v] = ForwardBatch(X_train(:,1+num_images*(i-1):num_images*i), W, b);
    % Analytical batch
    [grad_W, grad_b] = BackwardBN(X_train(:,1+num_images*(i-1):num_images*i), Y_train(:,1+num_images*(i-1):num_images*i), P, W, h, s, s_norm,mu_, v,  lambda);
    
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