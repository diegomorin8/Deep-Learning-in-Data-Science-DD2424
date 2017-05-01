%% Intro
%
% As demanded in the lab instructions, I have placed all the different
% files into one single file (I would rather leave them in separate files,
% but if this eases the corrector's work so be it :) ). 
%
% The function assignment2() contains an execution of the training of the
% model using the combination that achieved the maximum score. 
%
% *Table of Contents*
%
% assignment2() - This is the main script
% MiniBatchGD - MiniBatch algorithm
% LoadBatch - Load the sets (training, validation, test)
% EvaluateClassifier - Perform the forward pass
% ComputeCost - Compute the cost of the model on a data set
% ComputeAccuracy - Compute the accuracy of the model on a data set
% ComputeGradients - Compute the gradient (analytically)
% GradientCheck - Perform the gradient check (compare analytical and
%                   numerical)
% VisualizeLoss - Visualization of the Loss function (Training and
%                   Validation sets)
% VisualizeClassTemplates - Visualization of the class templates 
% ComputeGradsNum - Numerical implementation of the gradient computation
% ComputeGradsNumSlow - More precise numerical method
%


%% assignment1: Code for the assignment 2
function assignment2()
%ASSIGNMENT1   This function contains the code to load the datasets and
%train a model to achieve an accuracy above 50% on the test set
% [ Wstar, bstar] = assignment1( ensemble ) trains a model on a training 
% set and evaluates its accuracy on a test set
%    
% Inputs:
%   ensemble: Set to 0 if no ensemble is to be performed, set to 1
%   otherwise

% Clear workspace, load dataset directory
clear
%clc
addpath Datasets/cifar-10-batches-mat/;

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

% Center data using the mean of the training set
mu = mean(X_train, 2);
X_train = bsxfun(@minus, X_train, mu);
X_val = bsxfun(@minus, X_val, mu);
X_test = bsxfun(@minus, X_test, mu);

% Number of features (input layer)
[d, ~] = size(X_train);
% Number of neurons in the hidden layer (SET TO 750 for best result,
% spoiler alert: IT IS VERY SLOW)
m = 500;%750;%500;
% Number of classes (output layer)
[K, ~] = size(Y_train);
% Network structure
neuron_layers = [d, m, K];

% Number of samples
n_samples = size(X_train,2);

% Initialize model parameters
[W, b] = initParams(neuron_layers, n_samples);

% Initialize learning parameters
GDparams.n_batch = 100;%500;
eta = 8.548e-3;
GDparams.n_epochs = 100;
GDparams.rho = 0.95; % 0.9
GDparams.decay_rate = 0.95; %.99;
% Regularization term
lambda = 1.558e-5;%1e-4;%9e-5;%1.558e-5; 
% Noise added in training data
std_noise = 0;%sqrt(2/(100*n_samples));

% Check correctness of the Gradient implementation (No message should be
% displayed)
% GradientCheck(X_train, Y_train, W, b, lambda);


% Run Mini-batch SGD algorithm
[ Wstar, bstar, loss_train , loss_val] = MiniBatchGD( ...
    X_train, Y_train, y_train, X_val, Y_val, GDparams, eta, W, b, ...
    lambda, X_test, y_test, std_noise );

% Obtain accuracy on test data
acc = ComputeAccuracy( X_test, y_test, Wstar, bstar );
fprintf('Accuracy = %.2f %%\n', acc*100);

% Visualize Loss
VisualizeLoss(loss_train, loss_val);

end


%% MiniBatchGD: Mini-batch algorithm
function [ Wstar, bstar, loss_train , loss_val] = MiniBatchGD( X_train, ... 
    Y_train, y_train, X_val, Y_val, GDparams, eta, W, b, lambda, X_test, y_test,...
    std_noise )

% MINIBATCHGD  Implementation of the mini-batch gradient descent algorithm
%
% [ Wstar, bstar, loss_train , loss_val] = MiniBatchGD(X_train, ... 
%    Y_train, X_val, Y_val, GDparams, W, b, lambda, std_noise) performs the
%    mini-batch SGD updating the model parameters W and b based on the cost
%    computed on the training set X_train.
%
% Inputs:
%   X_train: Each column of X corresponds to an image, it has size (dxN).
%               Samples belong to train set.
%   Y_train: One-hot ground truth label for the corresponding image vector 
%           in X, it has size (KxN). Samples belong to train set.
%   X_val: Each column of X corresponds to an image, it has size (dxN)
%               Samples belong to validation set.
%   Y_val: One-hot ground truth label for the corresponding image vector 
%           in X, it has size (KxN). Samples belong to validation set.
%   GDparams: Parameters of the training
%   W: Weight matrix, it has size (Kxd)
%   b: Bias vector, it has size (Kx1)
%   lambda: Weight on the regularization term
%   std_noise: Standard deviation of the noise added to the training images
%
% Outputs:
%   Wstar: Optimal solution found for W, size (Kxd)
%   bstar: Optimal solution found for b, size (Kx1)
%   loss_train: Loss obtained on the training set
%   loss_val: Loss obtained on the validation set


% Obtain training parameters
[~, N] = size(X_train);
n_batch = GDparams.n_batch;
%eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
rho = GDparams.rho;
decay_rate = GDparams.decay_rate;

% Initialize loss on the training set
loss_train = zeros(n_epochs+1,1);
loss_train(1) = ComputeCost( X_train, Y_train, W, b, lambda );
fprintf('Cost = %d\n', loss_train(1));
% Initialize loss on the validation set
loss_val = zeros(n_epochs+1,1);
loss_val(1) = ComputeCost( X_val, Y_val, W, b, lambda );

n_hidden = numel(W);

% Define and initialize momentum cell
v = cell(2, n_hidden);
for k = 1:n_hidden
    v{1, k} = 0;
    v{2, k} = 0;
end

% Iterate for each epoch
for epoch=1:n_epochs
    rand_perm = randperm(size(X_train,2));
    X_train = X_train(:, rand_perm);
    Y_train = Y_train(:, rand_perm);
    y_train = y_train(rand_perm);
    
    % For each batch of data
    for j=1:N/n_batch
        
        % Obtain batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);
        
        % Add some jitter to pictures
        Xbatch = Xbatch + std_noise*randn(size(Xbatch));
        
        % Forward pass
        [P, S, H] = EvaluateClassifier( Xbatch, W, b );
        
        % Backward pass
        [db, dW] = ComputeGradients( Xbatch, Ybatch, P, S, H, W, lambda );

        % Update network parameters
        for k = 1:n_hidden
            % Momentum update
            v{1, k} = rho * v{1, k} + eta * db{k};
            v{2, k} = rho * v{2, k} + eta * dW{k};
            % Gradient update
            b{k} = b{k} - v{1, k};
            W{k} = W{k} - v{2, k};
        end
        
    end

    % Decrease learning rate
    eta = decay_rate*eta;
    
    % Obtain loss for training set
    loss_train(epoch+1) = ComputeCost( X_train, Y_train, W, b, lambda );
    % Check if the GD produces a stable update
%     if loss_train(epoch+1) >= 3*loss_train(1) || isnan(loss_train(epoch+1))
%         fprintf('Unstable! You might want to decrease the learning rate\n');
%         break;
%     end
    fprintf('%.d) Cost = %d\n', epoch, loss_train(epoch+1));
    
    % Obtain loss for validation set
    loss_val(epoch+1) = ComputeCost( X_val, Y_val, W, b, lambda );

    % Obtain accuracy in test set
    acc = ComputeAccuracy( X_test, y_test, W, b );
    fprintf('%.d) Accuracy = %.2f \n', epoch, acc*100);

    if acc == 1
        fprintf('It took %d epochs to reach perfection! \n', epoch);
        break;
    end
end

% Output results
Wstar = W;
bstar = b;

end

 %% initParans
function [W, b] = initParams(neurons_layer, n)
%INITPARAMS initializes the model parameters
%   [W, b] = initParams(neurons_layer, n)
%
% Inputs:
%   neurons_layer: vector containing neurons in each of the layers of the 
%                   network
%   n_samples: Number of samples
%
% Outputs:
%   W: Cell array containing the weight matrix of each layer
%   b: Cell array containing the bias vector of each layer


% Number of hidden layers
    n_hidden = numel(neurons_layer)-1;
    % Define cells for weight matrices and bias vectors
    W = cell(1,n_hidden);
    b = cell(1,n_hidden);
    % Standard deviation for weight matrices initialization, as proposed by
    % He et al. (201X)
    std_dev = sqrt(2.0/n);
    
    % Initialize each vector & matrix pair (for each layer)
    for i = 1:n_hidden
        rng(400);
        W{i} = std_dev*randn(neurons_layer(i+1), neurons_layer(i));
        b{i} = zeros(neurons_layer(i+1), 1); %std_dev*randn(neurons_layer(i+1), 1);
    end
end

%% LoadBatch
function [ X, Y, y ] = LoadBatch( filename )
%LOADBATCH Obtains the pixel values, label and one-hot-encoded label for a
%batch of images
%   [X, Y, y ] = LoadBatch( filename )
%
% Inputs:
%   filename: Name of the file to load

A = load(filename);

% Pixel data, size dxN (d: #features, N: #samples)
X = double(A.data)'/255;

% Label data, size N
y = double(A.labels+1);

% One-hot encoding of the label data, size KxN (K: #Classes)
Y = full(ind2vec(y'));
end

%% ComputeAccuracy
function [ acc ] = ComputeAccuracy( X, y, W, b )
% COMPUTEACCURACY  Computes the accuracy of the model on some dataset
%   acc = COMPUTEACCURACY(X, y, W, b) computes accuracy of the model
%   described by W and b on the set X with labels y.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   y: Ground truth labels for the corresponding image vectors in X,
%       it has size (nx1)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   acc: Accuracy obtained with the current model

% Obtain number of samples
n = size(X,2);

% Obtain scores
[P, ~, ~] = EvaluateClassifier( X, W, b );

% Obtain classification for each point
[~, idx] = max(P);

% Obtain accuracy
acc = sum(y == idx')/n;

end

%% ComputeCost
function [ J ] = ComputeCost( X, Y, W, b, lambda )
%COMPUTECOST Computes the cost function for a set of images
%   J = ComputeCost( X, Y, W, b, lambda ) computes the cost on the set X 
%   with labels y of the model described by parameters W and b, where 
%   lambda is the regularization strength.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   Y: One-hot ground truth label for the corresponding image vector in X,
%       it has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%   lambda: Weight on the regularization term
%   loss: Loss to be used, either softmax ("soft") or SVM ("svm")
%
% Outputs:
%   J: Cost obtained after adding the loss of the network predictions for 
%       images in X. It is a (scalar)

% Obtain class probabilities of each sample in X
[P, ~, ~] = EvaluateClassifier( X, W, b );

% Obtain the regularization term
reg = 0;

for i = 1:numel(W)
    reg = reg + sumsqr(W{i});
end

% Combine regularization term with cros-validation loss
D = size(X,2);

J = -1/D *sum(log(sum(Y.*P,1))) + lambda*reg;


end

%% EvaluateClassifier
function [ P, S, H ] = EvaluateClassifier( X, W, b )
% EVALUATECLASSIFIER   evaluates the scores of a batch of images
%   P = EvaluateClassifier( X, W, b ) performs the forward pass computing
%   the scores of each class for all data samples in X using the model
%   parameters W and b.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)
%   S: contains the scpres for each label (class) for each image 
%       in the columns of X. It has size (Kxn)
%

n_hidden = numel(W);

S = cell(1, n_hidden);
H = cell(1, n_hidden-1);

% Iteratively multiply the data matrix X by the matrix of Weights W and add 
% the bias b (broadcasting)
S{1} = bsxfun(@plus,W{1}*X,b{1});

for i =2:n_hidden
    H{i-1} = max(0, S{i-1});
    S{i} = bsxfun(@plus,W{i}*H{i-1},b{i});
end

P = bsxfun(@rdivide, exp(S{n_hidden}), sum(exp(S{n_hidden}), 1));

end

%% GradientCheck: This checks that the analytical Gradient is correctly implemented
function GradientCheck(X_train, Y_train, W, b, lambda )
%GRADIENTCHECK Verifies that the analytical expression of the Gradient is
%   correct.
%
%   GradientCheck(X_train, Y_train, W, b, lambda )
%
% Inputs:
%   X_train: Each column of X corresponds to an image, it has size (dxn)
%   Y_train: One-hot ground truth label for the corresponding image vector 
%       in x_train, it has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)

% We implement, analytically, the gradient computation. To verify that it
% is well implemented we run this "gradient check" test, which compares our
% results with the results obtained using numerical methods.

n_hidden = numel(W);

err_W = cell(n_hidden, 1);
err_b = cell(n_hidden, 1);

disp('Checking Gradient...');
X_train = X_train(1:100,:);
W{1} = W{1}(:,1:100);

% Runfor each batch
for i=1:100
    X_low = X_train(:,1+100*(i-1):100*i);
    Y_low = Y_train(:,1+100*(i-1):100*i);
    [P, S, H] = EvaluateClassifier( X_low, W, b );
    [ grad_b, grad_W ] = ComputeGradients( X_low, Y_low, P, S, H, W, ...
        lambda );
    [ ngrad_b, ngrad_W ] = ComputeGradsNumSlow(X_low, Y_low, W, b, ...
        lambda, 1e-6);
    
    for j = 1:n_hidden
        err_W{j} = norm(grad_W{j}(:)-ngrad_W{j}(:))/...
        (norm(grad_W{j}(:))+norm(ngrad_W{j}(:)));
        err_b{j} = norm(grad_b{j}-ngrad_b{j})/(norm(grad_b{j})+norm(ngrad_b{j}));
        
        % Display warning if the difference is above a threshold
        if (err_W{j}>1e-8)
            fprintf('Weight gradient error of %d in layer %d! \n', err_W{j}, j);
        end
        if (err_b{j}>1e-8)
            fprintf('Bias gradient error of %d in layer %d! \n', err_b{j}, j);
        end

    end
    
    
end

disp('Gradient checked!');
end

%% ComputeGradient
function [ grad_b, grad_W ] = ComputeGradients( X, Y, P, S, H, W, lambda )
%COMPUTEGRADIENTS Computes the gradients of the model parameters (W, b) for
%a mini-batch and using the softmax loss.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   Y: One-hot ground truth label for the corresponding image vector in X,
%       it has size (Kxn)
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   lambda: Weight on the regularization term
%   loss: Loss to be used, either softmax ("soft") or SVM ("svm")
%
% Outputs:
%   grad_W: Gradient of the Weight matrix, size (Kxd)
%   grad_b: Gradient of the bias vector, size (Kx1)

% Number of hidden layers
n_hidden = numel(W);
% Initialize outputs
grad_W = cell(1, n_hidden);
grad_b = cell(1, n_hidden);
% Merge X and hidden layers
X = [X H];

% Initialize gradients
for j=n_hidden:-1:1
    grad_W{j} = zeros(size(W{j}));
    grad_b{j} = zeros(1, size(W{j},1));
    %grad_b{j} = zeros(size(W{j},1), 1);
end

% Size of the batch
B = size(X{1}, 2);

for i=1:B
    y = Y(:,i);
    p = P(:, i);
    g = - y'/(y'*p) * (diag(p) - p*p');

    % For all layers (Propagate term g)
    for j=n_hidden:-1:1
        % Add gradient w.r.t. to model parameters for 
        grad_b{j} = grad_b{j} + g;
        grad_W{j} = grad_W{j} + g'*X{j}(:, i)';
        if j ~= 1
            g = g*W{j}*diag(S{j-1}(:, i)>0);
        end
    end
end



for j=n_hidden:-1:1
    % Normalize and add regularization term
    grad_W{j} = grad_W{j}/B + 2*lambda*W{j};
    grad_b{j} = grad_b{j}'/B;
end

end

%% ComputeGradientNumSlow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end

end

%% VisualizeLoss: Visualization of the loss (on training and validation)
function VisualizeLoss(loss_train, loss_val)
% COMPUTEACCURACY  Computes the accuracy of the model on some dataset
%   acc = COMPUTEACCURACY(X, y, W, b) computes accuracy of the model
%   described by W and b on the set X with labels y.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   y: Ground truth labels for the corresponding image vectors in X,
%       it has size (nx1)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   acc: Accuracy obtained with the current model


figure;
    plot(loss_train);
    hold on;
    plot(loss_val);
    h_legend = legend('Training loss', 'Validation loss');
    set(h_legend, 'Fontsize', 16, 'Interpreter','latex');
    set(gca,'fontsize',14)
    ylabel('Loss','Interpreter','latex', 'fontsize', 18);
    xlabel('Epoch','Interpreter','latex', 'fontsize', 18);
    grid on
end
